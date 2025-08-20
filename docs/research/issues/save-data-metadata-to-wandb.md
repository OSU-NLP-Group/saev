# Issue: Save Data Metadata to Wandb Runs

Look in src/saev/interactive/metrics.py to see the logic for looking at SAE training data from wandb.ai. Specifically, make_df() and find_metadata().

See how this notebook needs to be run on the same machine where the shards are located? That's a pain in the ass, especially if the metadata is deleted. 

Since the data we load from the metadata is tiny (just a JSON object), we could also just log it to wandb directly during the training process in train.py.

You need to:

1. Explore the different options for logging this sort of data to WandB. I think we could modify the config object directly before we init the run on WandB. There might also be other good ways to log this sort of data. I expect some web searches to be necessary. I think logging it as the config object is probably the best option, unless there is a massively more appealing option.
2. Figure out how to log that metadata to wandb in whatever way you decide. This metadata is probably available on the dataloader.
