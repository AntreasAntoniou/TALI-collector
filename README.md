
To use this repository, install the dependencies under install_dependencies.sh and then run the following command:

```bash
python3 collect_wit_tali_with_clip_filter.py --target_directory /mnt/nas/datasets/tali-wit/ --res 360p --num_threads 16 --sleep_duration 0.0 --pool_type Process --starting 0 --ending 100000 --clip_cutoff 0.5
```