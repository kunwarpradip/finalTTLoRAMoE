import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import tensorly as tl
import math, sys
import tensorly as tl
import torch.nn.init as init
from tensorly.decomposition import tensor_train

tl.set_backend('pytorch') #used to set the backend for the tensorly library to PyTorch

import torch
import tensorly as tl
from tensorly.decomposition import tensor_train

tl.set_backend('pytorch')

def tensorized_multiplication(X, tt_cores, m_factors, n_factors, device):

    B = X.size(0)
    S_len = X.size(1)

    # 1) Reshape X => [B, m1, m2, ...]
    shape_x = [B]+[S_len] + m_factors[::-1]
    tt_state = X.view(shape_x)  # e.g. [B, 4, 3] if m_factors=[4,3]

    # 2) Insert an initial rank dimension => [B, 1, m1, m2, ...]
    tt_state = tt_state.unsqueeze(1)  # shape [B, r=1, m1, m2, ...]

    # We'll do:
    #   - first "len(m_factors)" cores:  contract out each input factor
    #   - next "len(n_factors)" cores:   add each output factor dimension
    num_m = len(m_factors)
    num_n = len(n_factors)
    total_cores = num_m + num_n

    if len(tt_cores) != total_cores:
        raise ValueError(f"Expected {total_cores} TT-cores, got {len(tt_cores)}")

    # 3) Process each INPUT factor
    # We want to remove the last dimension from tt_state each time:
    # eq = 'b r ... m, r m p -> b p ...'
    # Explanation:
    #   - 'm' is the last dimension
    #   - '...' lumps any leftover dims in between 'r' and 'm'
    #   - we sum over (r, m), leaving 'b' and leftover '...' plus new rank p
    for i in range(num_m):
       
        core = tt_cores[i]  # shape [r_in, m_i, r_out]
        # base_core = layer_weight_cores[i]
        # core = base_core + alpha * initial_core
        # print(core.shape,tt_state.shape)
        eq_in = 'b r ... m, r m p -> b p ...'
        tt_state = torch.einsum(eq_in, tt_state, core)
        # shape now has same # of dims, except the "m" dimension is contracted out
        # and rank dimension might have changed from r_in to r_out

    # 4) Process each OUTPUT factor
    # Now each output factor is appended at the end:
    # eq = 'b r ..., r n p -> b p ... n'
    # Explanation:
    #   - we sum over 'r'
    #   - leftover dims remain in '...'
    #   - new factor dimension 'n' is appended at the end
    start_n = num_m
    for i in range(num_n):
        core = tt_cores[start_n + i]  # shape [r_in, n_i, r_out]
        # base_core = layer_weight_cores[start_n + i]
        # core = base_core + alpha * initial_core
        eq_out = 'b r ..., r n p -> b p ... n'
        tt_state = torch.einsum(eq_out, tt_state, core)
        # shape now has one more dimension at the end for 'n'

    # 5) Flatten to [B, -1] => [B, prod(n_factors)]
    Z = tt_state.view(B,S_len, -1)
    return Z



class AdaptCores_and_Test_Individual(nn.Module): #Inherits from nn.Module
        def __init__(self, 
                    module: nn.Linear,
                    alpha: int,
                    m_factors: list,
                    n_factors: list,
                    layer_idx, 
                    ttlora_cores: list, 
                    device,
                    ):
            super().__init__()
            self.base_module = module
            self.alpha=alpha
            self.m_factors = m_factors
            self.n_factors = n_factors
            self.layer_idx = layer_idx
            self.ttlora_cores = ttlora_cores
            self.device = device

        def forward(self, x: torch.Tensor) -> torch.Tensor: # x is input used in forward pass at every call of model
            if self.alpha > 0:
                # Ensure all tensors are on the same device
                out = tensorized_multiplication(x.to(self.device), 
                                                    tt_cores = self.ttlora_cores, 
                                                    m_factors=self.m_factors, 
                                                    n_factors=self.n_factors, 
                                                    device=self.device,) 
                # print("Executed query")
                return self.base_module(x.to(self.device)) + out*self.alpha

class TTLoRALinearWrapper_withcores(nn.Module): #Inherits from nn.Module
        '''Define cores and the forward pass and makes it ready for training the cores'''
        def __init__(self, module: nn.Module, tt_shape, tt_rank, alpha:int, m_factors, n_factors, device, init_choice):
            super().__init__()
            self.base_module = module
            self.tt_shape = tt_shape
            self.tt_rank = tt_rank
            self.alpha=alpha
            self.m_factors = m_factors
            self.n_factors = n_factors
            self.device = device
            self.init_choice = init_choice
            self.in_features_shape, self.out_features_shape = self.base_module.weight.shape
            
            if self.init_choice == "direct_init":
                self.tt_cores = self.generate_cores(self.tt_shape, self.tt_rank).to(self.device)  # Change method as needed
                self.tt_cores.requires_grad= True 
                # Make the bias non-trainable
                if self.base_module.bias is not None:
                        self.base_module.bias.requires_grad = False
            
            elif self.init_choice == "init_and_decompose":
                '''Create a torch tensor dummy Weight_delta of shape (in_feature_shape, out_feature_shape) 
                and initialize all 0s'''
                self.Weight_delta=torch.zeros((self.in_features_shape, self.out_features_shape)).to('cuda')
                '''Then allocate random values using gaussian distribution to dummy Weight_delta'''
                self.reset_parameters()
                '''Decompose the dummy Weight_delta to high dimensional tensor based on the TT shapes'''
                self.Weight_TT_dimension = self.reshape_tensor(torch.tensor(self.Weight_delta)).to('cuda')
                '''We have dummy weight decomposed into multiple tensors based on tt_shape
                Now, we create tensor cores as Parameters which are trainable
                Paramerter wraps the tensors into traninable parameters
                ParameterList holds the list of parameters
                TT Cores are initialized using standard normal distribution based on the ttcores shapes'''
                self.tt_cores = nn.ParameterList([nn.Parameter(self.initialize_cores(*shape).to('cuda')) for shape in self.get_ttcores_shapes()])
                '''Using tensor train, decompose into multiple tensors based on the ranks and shapes provided'''
                self.tt_cores_dummy = tensor_train(self.Weight_TT_dimension, self.tt_rank)
                '''Transfer the values of tensor trained ttlora_cores_dummy to ttlora_cores trainable parameters'''
                for i in range(len(self.tt_cores)):
                    self.tt_cores[i].data = torch.tensor(self.tt_cores_dummy[i], dtype=torch.float32).to('cuda')
            
                self.tt_cores.requires_grad= True 
                # Make the bias non-trainable
                if self.base_module.bias is not None:
                        self.base_module.bias.requires_grad = False
            else:
                raise ValueError("Invalid initialization choice")

        def generate_cores(self, shape, rank):
            # print("shape",shape)
            # print("rank",rank)
            # sys.exit()
            tt_cores = nn.ParameterList()  # Store TT cores as trainable parameters

            for i in range(len(shape)):
                core_shape = (rank[i], shape[i], rank[i + 1])  # TT core shape
                core = torch.empty(core_shape)  # Create empty tensor
            
                tt_cores.append(nn.Parameter(core))  # Store as a trainable parameter
            
            for i in range(len(tt_cores)):
                    nn.init.kaiming_uniform_(tt_cores[i], a=math.sqrt(8))
                    tt_cores[i].data /= (tt_cores[i].data.norm() + 1e-6)  # Normalize cores

            return tt_cores 
        
        def get_ttcores_shapes(self):
            shapes = []
            ranks = self.tt_rank
            for i in range(len(self.tt_shape)):
                shape = (ranks[i], self.tt_shape[i], ranks[i + 1])
                shapes.append(shape)
            return shapes

        def reshape_tensor(self, tensor):
            return tensor.reshape(*self.tt_shape) ## * unpacks the tt_shape list into individual arguments

        def reset_parameters(self):
            '''Initialize the given tensor with random values from a gaussian distribution'''
            torch.manual_seed(42)
            nn.init.kaiming_uniform_(self.Weight_delta, a=math.sqrt(8))
        
        def reset_parameters(self):
            '''Initialize the given tensor with random values from a gaussian distribution'''
            torch.manual_seed(42)
            nn.init.kaiming_uniform_(self.Weight_delta, a=math.sqrt(8))

        def initialize_cores(self, *shape):
            '''Initialize the given tensor with random values from a standard normal distribution (mean = 0 and std = 1)
            and scaled by a calculated standard deviation'''
            std = 1.0 / math.sqrt(shape[1]) #Standard deviation
            return torch.randn(*shape) * std
            
        def forward(self, x: torch.Tensor) -> torch.Tensor: # x is input used in forward pass at every call of model
            
            if self.alpha > 0:
                out = tensorized_multiplication(x.to(self.device), 
                                                self.tt_cores, 
                                                m_factors=self.m_factors, 
                                                n_factors=self.n_factors, 
                                                device=self.device) 

                return self.base_module(x.to(self.device)) + out*self.alpha