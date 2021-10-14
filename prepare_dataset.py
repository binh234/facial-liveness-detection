import os
import pandas as pd
import itertools
import random
import requests
from sklearn.model_selection import train_test_split
from io import BytesIO
from zipfile import ZipFile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def unzip(zip_file, extract_to='.'):
    zipfile = ZipFile(zip_file)
    zipfile.extractall(path=extract_to)


def pos_pairing(lst):
    ret = []
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            if lst[j] == lst[i]:
                pass
            else:
                ret.append((lst[i], lst[j]))
    return ret


def pos_pairing2(lst, lst2):
    return [(x, y) for x in lst for y in lst2 if not x == y]


def neg_pairing(lst, lst2):
    return [(x, y) for x in lst for y in lst2]


def generate_samples(real_path, fake_path, real_sub_dir, df, num_pairs=2500):
    for directory in real_sub_dir:
        pos_images = [name for name in os.listdir(
            f'{real_path}/{directory}') if not (name.startswith(".") or name.startswith("Thumb"))]
        neg_images = [name for name in os.listdir(
            f'{fake_path}/{directory}') if not (name.startswith(".") or name.startswith("Thumb"))]
        pos_pairs = list(itertools.combinations(pos_images, 2))

        if len(pos_pairs) > num_pairs:
            pos_pairs = random.choices(pos_pairs, k=num_pairs)

        for i in pos_pairs:
            new_row = {'sample': directory,
                       'image1': f'{real_path}/{directory}/{i[0]}', 'image2': f'{real_path}/{directory}/{i[1]}', 'label': 1}
            df = df.append(new_row, ignore_index=True)

        neg_pairs = list(itertools.product(pos_images, neg_images))
        if len(neg_pairs) > num_pairs:
            neg_pairs = random.choices(neg_pairs, k=num_pairs)
        for i in neg_pairs:
            new_neg_row = {
                'sample': directory, 'image1': f'{real_path}/{directory}/{i[0]}', 'image2': f'{fake_path}/{directory}/{i[1]}', 'label': 0}
            df = df.append(new_neg_row, ignore_index=True)

    return df


DATASET_ID = "1oE6yv-RYV5_4HDjUo6F8mJofzrW_tdTo"


def prepare_dataset(data_folder="ds"):
    """
    Prepare training dataset
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    # Download dataset
    zip_file = "./Detectedface.zip"

    download_file_from_google_drive(DATASET_ID, zip_file)
    unzip(zip_file, data_folder)

    if not os.path.exists(os.path.join(data_folder, "Detectedface")):
        print("Can't download dataset")
        return

    real_path = os.path.join(data_folder, "Detectedface", "ClientFace")
    fake_path = os.path.join(data_folder, "Detectedface", "ImposterFace")

    os.rename(data_folder + "/Detectedface/ClientFace/0013", data_folder + "/Detectedface/ClientFace/0016")

    real_sub_dir = [name for name in os.listdir(
        real_path) if not name.startswith(".")]
    real_sub_dir.sort()
    fake_sub_dir = [name for name in os.listdir(
        fake_path) if not name.startswith(".")]
    fake_sub_dir.sort()

    column_names = ["sample", "image1", "image2", "label"]

    df = pd.DataFrame(columns=column_names)
    test_df = pd.DataFrame(columns=column_names)

    # Take last 4 clients for testing, others for training and validation
    print("Generating training samples..")
    df = generate_samples(real_path, fake_path, real_sub_dir[:-4], df)
    print("Generating testing samples..")
    test_df = generate_samples(
        real_path, fake_path, real_sub_dir[-4:], test_df)

    # UNCOMMENT TO SAVE CSV
    test_df.to_csv('./dataset/test_data.csv', encoding='utf-8', index=False)
    y = df.label
    X = df.drop('label', axis=1)
    df.reset_index(drop=True, inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)

    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    X = pd.concat([X_train, y_train], axis=1)
    X_val = pd.concat([X_val, y_val], axis=1)

    X.to_csv('./dataset/train_data.csv', encoding='utf-8', index=False)
    X_val.to_csv('./dataset/val_data.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    prepare_dataset()
