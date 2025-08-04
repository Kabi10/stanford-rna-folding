# Pushing Your Work to Kaggle

This guide explains how to use the `push_to_kaggle.py` script to publish your local code to Kaggle as notebooks or kernels.

## Prerequisites

- Kaggle API credentials set up (see [KAGGLE_SETUP.md](KAGGLE_SETUP.md))
- Python 3.6+ installed
- Required dependencies installed (`pip install -r requirements.txt`)

## Basic Usage

The script provides two main commands:

### 1. Push the Titanic Model

To push the Titanic model to Kaggle:

```bash
python scripts/push_to_kaggle.py titanic
```

This will:
- Create a kernel with the title "Titanic Survival Prediction Model"
- Associate it with the Titanic competition
- Set it as private by default (only you can see it)

Optional arguments:
- `--title "My Custom Title"` - Use a custom title
- `--public` - Make the kernel public (visible to everyone)

Example:
```bash
python scripts/push_to_kaggle.py titanic --title "My Advanced Titanic Model" --public
```

### 2. Push Any Code File

You can also push any Python script or Jupyter notebook to Kaggle:

```bash
python scripts/push_to_kaggle.py push <file_path>
```

Optional arguments:
- `--title "My Kernel Title"` - Specify the kernel title (otherwise derived from filename)
- `--competition <competition_name>` - Associate with a competition
- `--dataset <dataset_name>` - Associate with a dataset
- `--public` - Make the kernel public

Example:
```bash
python scripts/push_to_kaggle.py push path/to/my_script.py --competition titanic --public
```

## Auto-Updates

After pushing your code to Kaggle, you can update it by simply running the same command again after making changes locally. Kaggle will update the existing kernel rather than creating a new one.

## Viewing Your Kernels

To view all your kernels on Kaggle:

1. Log in to [Kaggle](https://www.kaggle.com)
2. Go to your profile
3. Click on the "Code" tab

You can also run this command to list your kernels:
```bash
kaggle kernels list --mine
```

## Troubleshooting

If you encounter issues:

1. **Authentication errors**: Make sure your Kaggle API credentials are set up correctly
2. **"Kernel already exists"**: This is normal - your kernel will be updated
3. **File not found**: Check the path to your code file
4. **Push fails**: Check your internet connection and try again

## Advanced: Converting Between Script and Notebook

The Kaggle API can convert Python scripts (.py) to notebooks (.ipynb) automatically. If you prefer working with notebooks on Kaggle:

1. Push your Python script
2. On Kaggle, click "Copy & Edit"
3. Select "Notebook" as the kernel type

## Next Steps

After pushing your code to Kaggle:

1. Run your kernel on Kaggle to verify it works
2. Make any necessary adjustments directly on Kaggle or locally
3. Submit competition predictions by adding code to save submission files
4. Share your work with the community if you made it public 