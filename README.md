
To use this repository, install the dependencies under install_dependencies.sh and then run the following command:

```bash
python3 collect_wit_tali_with_clip_filter.py --target_dataset_dir /mnt/tali-wit-part-0/tali-wit-2.1/ --res 360p --num_threads 96 --sleep_duration 0.0 --pool_type Process --starting 0 --ending 100000 --clip_cutoff 0.5 --wit_cache_dir /mnt/tali-wit-part-0/wit/
```
