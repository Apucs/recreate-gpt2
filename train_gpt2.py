import os
import time
import inspect
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

#--------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #not really a bias, more of a mask but following the OpenaAI/HF naming though
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B,T,C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        #attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flashed attention==> lot faster than the above four lines combined

        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) --> (B, T, C) re-assemble all head outputs side by side 
        #output projection
        y = self.c_proj(y)
        return y 

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    # block_size: int = 256
    # vocab_size: int = 65
    # n_layer: int = 6
    # n_head: int = 6
    # n_embd: int = 384

    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding size

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing scheme
        # the word embedding and the final linear layer share the same weights
        # this is a common technique to reduce the number of parameters in the model
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer**-0.5)
            # initialize linear layers with a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # initialize embeddings with a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B,T) where B is the batch size and T is the sequence length
        # we need to get the position embeddings for each token in the sequence
        # and add them to the token embeddings
        # idx is the input tensor of shape (B,T)
        # wte is the word embedding tensor of shape (vocab_size, n_embd)
        # wpe is the position embedding tensor of shape (block_size, n_embd)
        # wte(idx) is the word embedding of shape (B,T,n_embd)
        # wpe(idx) is the position embedding of shape (B,T,n_embd)
        # wte(idx) + wpe(idx) is the input tensor of shape (B,T,n_embd)
        # we need to transpose the input tensor to (T,B,n_embd) for the transformer
        # transformer is a ModuleDict of the transformer layers
        # h is a ModuleList of the transformer layers
        # ln_f is the final layer norm
        # lm_head is the final linear layer
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token embeddings and position embeddings through the transformer
        pos = torch.arange(T, dtype=torch.long, device=idx.device) # (T,)
        pos_emb = self.transformer.wpe(pos) # (T,n_embd)
        tok_emb = self.transformer.wte(idx) # (B,T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layer norm and the classification head
        x = self.transformer.ln_f(x) # (B,T,n_embd)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # (B*T, vocab_size) vs (B*T,)
        
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # print(f"ending with k: {k}")
                # print(f"sd_hf[k].shape: {sd_hf[k].shape}")
                # print(f"sd[k].shape: {sd[k].shape}")
                # print(f"sd_hf[k].shape[::-1]: {sd_hf[k].shape[::-1]}\n")

                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_deacy=0.1, learning_rate=6e-4, device='cpu'):

        # for pn, p in self.named_parameters():
        #     print(pn, p)

        # start with all of the candidate parameters (that require gradients)
        params_dict = {pn: p for pn, p in self.named_parameters()}
        params_dict = {pn: p for pn, p in params_dict.items() if p.requires_grad}
        # create optim groups, Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e, all weight tensors in mutuals + embeddings decay, all biases and layernomrs don't.
        decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in params_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_deacy},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused adamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer

# ------------------------------------------------------------

# get data batch
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1):
        self.B = B # batch size 
        self.T = T # sequence length
        self.process_rank = process_rank # rank of the current process
        self.num_processes = num_processes # total number of processes

        # at init time, load token from disk and store it in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)  # (N,)

        print(f"loaded {len(self.tokens)} tokens from input.txt")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = self.B*self.T * self.process_rank # start at the beginning of the tensor, but offset by the process rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1] # (B*T,)
        x = (buf[:-1].view(B, T)) # (B,T) input tokens
        y = (buf[1:].view(B, T)) # (B,T) target tokens

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes # advance by the batch size times the sequence length times the number of processes

        # if loading the next batch would go beyond the end of the tensor, reset to the beginning
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            print("reached end of tokens, resetting position to 0")
            self.current_position = self.B*self.T * self.process_rank
        
        return x, y

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
print(f"ddp: {ddp}")
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
    
total_batch_size = 524288
B = 4
T = 16
assert total_batch_size%(B*T*ddp_world_size)==0, f"make sure total_batch_size is divisible by B*T*ddp_world_size, got {total_batch_size} % ({B}*{T}*{ddp_world_size}) != 0"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # total number of gradient accumulation steps

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size) # batch size 4, sequence length 32

torch.set_float32_matmul_precision('high') # set the matmul precision to high for better performance

# create the model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    # wrap the model in DDP
    model = DDP(model, device_ids=[ddp_local_rank])
print(f"model compiled successfully....")

raw_model = model.module if ddp else model # unwrap the model if it is wrapped in DDP

max_lr = 6e-4 # max learning rate
min_lr = max_lr*0.1 # min learning rate
warmup_steps = 10 # number of warmup steps
max_steps = 50 # total number of steps to train for

def get_lr(it):
    # Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    # if it>lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"decay_ratio {decay_ratio} out of bounds"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starting at 1.0 and decaying to 0.0
    return min_lr + (max_lr - min_lr) * coeff # linear interpolation between min_lr and max_lr
    


# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps = 1e-8)
optimizer = raw_model.configure_optimizers(weight_deacy = 0.1, learning_rate = 6e-4, device=device) # configure the optimizer for the model
for step in range(max_steps):
    t0 = time.time()
    # x, y = train_loader.next_batch() # (B,T), (B,T)
    # x, y = x.to(device), y.to(device) # move to device
    
    optimizer.zero_grad()
    loss_accum = 0.0 # reset the loss accumulator
    
    # x, y = train_loader.next_batch() # (B,T), (B,T)
    # x, y = x.to(device), y.to(device) # move to device
    # logits, loss = model(x, y)
    # loss.backward()

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B,T), (B,T)
        x, y = x.to(device), y.to(device) # move to device
        
        # logits, loss = model(x, y) # (B,T,vocab_size), (B,)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # forward pass
            # logits is of shape (B,T,vocab_size), loss is of shape (B,)
            # logits contains the predicted probabilities for each token in the vocabulary
            # loss is the cross-entropy loss between the predicted and target tokens
            logits, loss = model(x, y) # (B,T,vocab_size), (B,)
            
        loss = loss / grad_accum_steps # scale the loss by the number of gradient accumulation steps
        loss_accum += loss.detach() # accumulate the loss
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1) # ensure that gradients are synchronized across all processes
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM) # sum the loss across all processes
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients to avoid exploding gradients

    #determine and set the learning rate for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # wait for all CUDA operations to finish
    t1 = time.time()
    dt = t1 - t0 # convert to seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size  # total number of tokens processed in this step
    tokens_per_sec = tokens_processed/dt # tokens per second

    if master_process:
        print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm : {norm:.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group() # clean up the process group
import sys; sys.exit(0)
#--------------------------------------------------------

# prefix tokens
model.eval()
max_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)
# model.to('cuda')
# model.to('cpu')
# print("didn't crash yay!")

tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1) # (5,8)
x = tokens.to(device)
# x = tokens.to('cuda')
# x = tokens.to('cpu')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B,T,vocab_size)
        # take the logits of the last position
        logits = logits[:, -1, :] # (B,vocab_size)
        probs = F.softmax(logits, dim=-1) # (B,vocab_size)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5,50), topk_indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B,1)
        # gather the corresponding indices from the top-k indices
        xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1) # (B,T+1)

# print the generated text
for i in range(max_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)