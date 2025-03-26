import torch.optim as optim

def get_optimizer(model, config):
    """
    모델 파라미터에 대한 optimizer를 생성합니다.
    
    config: dict, optimizer 설정 (예: optimizer 종류, learning rate, weight_decay 등)
      - optimizer: "AdamW", "Adam", "SGD" 등 (기본값 "AdamW")
      - lr: learning rate (기본값 1e-3)
      - weight_decay: (기본값 1e-5)
      - (SGD인 경우) momentum: (기본값 0.9)
    
    Returns:
      optimizer: torch.optim.Optimizer 객체
    """
    optimizer_name = config.get("optimizer", "AdamW")
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-5)
    
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer
