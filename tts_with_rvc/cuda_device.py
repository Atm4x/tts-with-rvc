def set_cuda_device_for_thread(torch_module, device) -> None:
    if not str(device).startswith("cuda:"):
        return
    selected_device = torch_module.device(device)
    torch_module.cuda.set_device(selected_device)
