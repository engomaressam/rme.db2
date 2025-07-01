"""
GPU Monitoring and Performance Tracking
---------------------------------------
This script helps monitor GPU performance and verify CUDA/GPU acceleration
for PyTorch models.
"""

def check_gpu_availability():
    """Check if CUDA is available and print GPU information."""
    import torch
    import GPUtil
    from tabulate import tabulate
    import time
    import psutil
    import platform
    import cpuinfo
    
    print("="*80)
    print("SYSTEM AND HARDWARE INFORMATION")
    print("="*80)
    
    # Basic system info
    print(f"\n{'System:':<20}{platform.system()} {platform.release()}")
    print(f"{'Python Version:':<20}{platform.python_version()}")
    print(f"{'PyTorch Version:':<20}{torch.__version__}")
    
    # CPU Info
    cpu_info = cpuinfo.get_cpu_info()
    print(f"\n{'CPU:':<20}{cpu_info['brand_raw']}")
    print(f"{'Physical Cores:':<20}{psutil.cpu_count(logical=False)}")
    print(f"{'Total Cores:':<20}{psutil.cpu_count(logical=True)}")
    
    # RAM Info
    ram = psutil.virtual_memory()
    print(f"\n{'Total RAM:':<20}{ram.total / (1024**3):.2f} GB")
    print(f"{'Available RAM:':<20}{ram.available / (1024**3):.2f} GB")
    
    print("\n" + "="*80)
    print("GPU INFORMATION")
    print("="*80)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Get GPU details
        gpus = GPUtil.getGPUs()
        gpu_list = []
        
        for i, gpu in enumerate(gpus):
            gpu_list.append([
                f"GPU {i}",
                gpu.name,
                f"{gpu.memoryTotal}MB",
                f"{gpu.memoryFree}MB",
                f"{gpu.memoryUsed}MB",
                f"{gpu.temperature}°C",
                f"{gpu.load*100}%"
            ])
        
        print("\n" + tabulate(gpu_list, 
                      headers=["Device", "Name", "Memory Total", "Memory Free", 
                              "Memory Used", "Temperature", "Load"],
                      tablefmt="grid"))
        
        # Set default device to first GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        # Additional CUDA info
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"CuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Current CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        
        # Test tensor operations
        print("\nTesting tensor operations...")
        x = torch.randn(10000, 10000)
        y = torch.randn(10000, 10000)
        
        # CPU operation
        start_time = time.time()
        _ = x @ y
        cpu_time = time.time() - start_time
        
        # GPU operation
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        
        # Warm-up
        _ = x_gpu @ y_gpu
        torch.cuda.synchronize()
        
        start_time = time.time()
        _ = x_gpu @ y_gpu
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"\nMatrix Multiplication (10000x10000):")
        print(f"CPU Time: {cpu_time*1000:.2f} ms")
        print(f"GPU Time: {gpu_time*1000:.2f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
    else:
        print("CUDA is not available. PyTorch is running on CPU.")
    
    print("\n" + "="*80)


def track_training(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    Track GPU memory usage and training performance.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
    """
    import torch
    import time
    from tqdm import tqdm
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot track GPU performance.")
        return
    
    device = torch.device("cuda:0")
    model = model.to(device)
    
    # Track metrics
    metrics = {
        'epoch_times': [],
        'batch_times': [],
        'memory_allocated': [],
        'memory_cached': []
    }
    
    print(f"\n{'='*80}")
    print("TRAINING PERFORMANCE METRICS")
    print(f"{'='*80}\n")
    
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        # Track batch times
        batch_times = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, 
                                                     desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch_start = time.time()
            
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Record memory usage
            metrics['memory_allocated'].append(torch.cuda.memory_allocated() / (1024**2))  # Convert to MB
            metrics['memory_cached'].append(torch.cuda.memory_cached() / (1024**2))  # Convert to MB
        
        # Record epoch time
        epoch_time = time.time() - epoch_start
        metrics['epoch_times'].append(epoch_time)
        metrics['batch_times'].extend(batch_times)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Time: {epoch_time:.2f}s, "
              f"Avg Batch Time: {sum(batch_times)/len(batch_times):.4f}s, "
              f"Memory Allocated: {metrics['memory_allocated'][-1]:.2f}MB")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total Training Time: {sum(metrics['epoch_times']):.2f}s")
    print(f"Average Epoch Time: {sum(metrics['epoch_times'])/len(metrics['epoch_times']):.2f}s")
    print(f"Average Batch Time: {sum(metrics['batch_times'])/len(metrics['batch_times']):.4f}s")
    print(f"Peak Memory Allocated: {max(metrics['memory_allocated']):.2f}MB")
    print(f"Peak Memory Cached: {max(metrics['memory_cached']):.2f}MB")
    
    return metrics
