# Eminence: Feature Boundary Ambiguity Backdoor Attack (KDD 2026 Official Release)
*The Eminence in Shadow: Exploiting Feature Boundary Ambiguity for Robust Backdoor Attacks, KDD 2026.*

> *“I AM ATOMIC.”*
> ***— Shadow***


**Eminence** is a backdoor attack framework that leverages **feature boundary ambiguity** to achieve high attack success under extremely low poisoning budgets, while preserving clean accuracy and enabling strong transferability across models.

This repository contains the source code, training scripts, trigger interface, and dataset loaders used in our KDD 2026 experiments.

---

## 🔥 Key Features
| Feature | Description |
|---|---|
| Ultra-low poison ratio ( < 0.1%) | High ASR while minimally perturbing training data |
| Boundary ambiguity optimization | Trigger embedding disrupts class separation |
| High transferability across architectures | Robust even when migrating to unseen models |
| Plug-and-play scripts | Reproduces results with a single run command |
| Full training & evaluation pipeline | Research-grade reproducibility |

---

## ⚙️ Installation

Dependencies:

+ Python >= 3.9
+ torch == 2.4.0 (**recommended**)

## 📂 Project Structure

```csharp
Enkidu/                   
├── core/                         # Core implementation    
│   ├── attacks/    
│       └── Eminence.py                 # Eminence Attack     
├── experiments/                   # Log                         
├── test_Eminence.py                # Example usage (Python API)                               
└── README.md                    
```

## 🚀 Usage

1. Python API:
   
   You can call Eminence directly inside Python scripts:

   ```python
   eminence = Eminence(
        train_dataset=trainset, # target dataset
        test_dataset=testset,
        model=core.models.ResNet(18, 10), # target model
        loss=nn.CrossEntropyLoss(),
        poison_ratio=0.0001,
        trigger_info={           # initialized trigger pattern
            'pattern': pattern,
            'weight': weight
        },
        label_mode='DIRTY', # label mode: 'CLEAN' or 'DIRTY'
        target_label=0,
        train_scale=0.3,
        optimize_model=surrogate_model,  # surrogate model
        optimize_dataset=surrogate_dataset, # surrogate dataset
        optimize_device=torch.device(f'cuda:{GPU}')
    )

    schedule = {
        'device': 'GPU',
        # 'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_SELECTED_DEVICES': GPU,
        'GPU_num': 1,

        'benign_training': False,
        'batch_size': 1024,
        'num_workers': 16,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'Eminence'
    }

    eminence.train(schedule)
    eminence.test(schedule)
   ```

## 📖 Citation

Soon

## Acknowledgements

This implementation is **built upon and modified from the BackdoorBox framework**, which offers a clean and well-structured design for backdoor attack research.  

We sincerely appreciate their contribution to the community:

🔗 [BackdoorBox GitHub Repository](https://github.com/THUYimingLi/BackdoorBox)

The primary author and maintainer of this repository is Zhou Feng, ZJU NESA Lab.  




If our work is helpful to your research, a citation or ⭐ on GitHub is appreciated.