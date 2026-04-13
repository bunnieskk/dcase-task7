## Structure of my project and data
```
/home/dcase_task7/cyq/dcase2026_task7_baseline/
├── baseline/                        # Main baseline code directory
│   ├── baseline_DIL_task7.py        # Main baseline execution script
│   └── domain_net.py                
├── checkpoints/BN/                  # Model weights saved after training
│   ├── checkpoint_D1.pth            # Domain 1 
│   ├── checkpoint_D2.pth            # Domain 2 
│   └── checkpoint_D3.pth            # Domain 3 
├── checkpoints_original/BN/         # Original pre-trained models; used to reproduce baseline performance
│   ├── checkpoint_D1.pth            # Domain 1
│   ├── checkpoint_D2.pth            # Domain 2
│   └── checkpoint_D3.pth            # Domain 3
├── results/                         
├── utils/                           
├── parameters_accuracy.txt          # Records the performance (accuracy) of models trained with different parameters
├── README.md                        
├── requirements.txt                 # Environment dependencies
└── task7_domain.png                 

/data/yunqichen/task7_data/
├── audio_chunked/                   # Audio data chunked into 4-second segments (used as model input)
│   ├── DIL-DCASE26-Dev-D2/          # Domain 2 (D2) development set data
│   │   ├── d2-dev-test/
│   │   └── d2-dev-train/
│   └── DIL-DCASE26-Dev-D3/          # Domain 3 (D3) development set data
│       ├── d3-dev-test/
│       └── d3-dev-train/
├── audio_original/                  # Original, unprocessed full-length audio files
│   ├── DIL-DCASE26-Dev-D2/
│   └── DIL-DCASE26-Dev-D3/
├── evaluation_setup/                
├── correction.py                    
├── development_train.txt            # Original audio training set split list and labels
├── development_test.txt             # Original audio test set split list and labels
├── development_train_chunked.txt    # Chunked audio training set split list and labels
└── development_test_chunked.txt     # Chunked audio test set split list and labels
```

## How to use
Before training, you should copy the original D1 model to a folder that is named /checkpoints/BN(just as my project's structure), since the data of Domain 1 is not provided.If you want to reproduce the performance of the baseline, you may copy the original D2 and D3 model at the same time, and use --resume instead of --save when running(according to the instruction in README.MD).

When running, you may use the command like
```
python baseline/baseline_DIL_task7.py train --augmentation='none' --learning_rate=1e-4 --batch_size=32 --cuda --num_workers 16 --epoch 120 --save
```
--augmentation='none' can be repalced by --augentation='mixup', as the script supports.

## Attention
If you want to run the script with data in your folder, don't forget to replace the path in files in /data/yunqichen/task7_data/evaluation_setup/ with your own path.