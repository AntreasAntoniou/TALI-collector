
To use this repository, install the dependencies under install_dependencies.sh and then run the following command:

```bash
python3 run.py --target_dataset_dir /mnt/data/tali-wit-2.1/ --res 360p --num_workers 32 --sleep_duration 0.1 --pool_type Process --starting 0 --ending 1000000 --clip_cutoff 0.3 --wit_cache_dir /mnt/data/wit/
```
