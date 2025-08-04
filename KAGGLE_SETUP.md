# Setting Up Kaggle API Access

This guide will help you set up Kaggle API credentials to interact with Kaggle programmatically.

## Prerequisites

- A Kaggle account (create one at [kaggle.com](https://www.kaggle.com) if you don't have one)
- Python and pip installed on your system
- Kaggle API package (installed via `pip install kaggle` or included in requirements.txt)

## Steps to Set Up Kaggle API Credentials

### 1. Generate API Token

1. Log in to your Kaggle account
2. Go to your account settings page by clicking on your profile picture in the top-right corner and selecting "Account"
3. Scroll down to the "API" section
4. Click "Create New API Token"
5. This will download a file named `kaggle.json` containing your API credentials

### 2. Configure API Credentials

#### Windows:

1. Create a folder named `.kaggle` in your user directory (`C:\Users\<YOUR_USERNAME>`)
2. Move the downloaded `kaggle.json` file to this folder
   ```
   move kaggle.json C:\Users\<YOUR_USERNAME>\.kaggle\
   ```

#### macOS/Linux:

1. Create a .kaggle directory in your home folder:
   ```bash
   mkdir -p ~/.kaggle
   ```
2. Move the downloaded kaggle.json file to this folder:
   ```bash
   mv kaggle.json ~/.kaggle/
   ```
3. Set appropriate permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Test Your Connection

To verify your API credentials are working correctly:

```bash
kaggle competitions list
```

This should display a list of active Kaggle competitions.

## Using the Kaggle API in This Workspace

Our workspace includes scripts to interact with Kaggle:

### Download Competition Data

Use the `download_kaggle_data.py` script to download competition data:

```bash
python scripts/download_kaggle_data.py competition <competition_name>
```

For example:
```bash
python scripts/download_kaggle_data.py competition titanic
```

### Download Dataset

To download a dataset:

```bash
python scripts/download_kaggle_data.py dataset <owner/dataset-name>
```

For example:
```bash
python scripts/download_kaggle_data.py dataset fedesoriano/heart-failure-prediction
```

## Common Issues and Solutions

### Permission Denied Error on Linux/macOS

If you encounter permission errors, ensure the kaggle.json file has the correct permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### API Token Not Found

If you see an error like "Kaggle API credentials not found", make sure:
- The kaggle.json file is in the correct location
- You've installed the Kaggle API package
- The JSON file has valid credentials

### Troubleshooting Script Issues

If our scripts aren't working correctly:
1. Ensure you've installed the requirements: `pip install -r requirements.txt`
2. Verify your API credentials are correctly set up
3. Make sure you're using the correct command syntax

## Additional Resources

- [Official Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle API Command Reference](https://www.kaggle.com/docs/api)
- [Kaggle API Python Client](https://www.kaggle.com/docs/api) 