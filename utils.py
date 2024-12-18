import json
from ollama import AsyncClient
from datasets import load_dataset, concatenate_datasets

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
CLIENT_OLLAMA = AsyncClient(
  host='http://192.168.1.5:11434',
)

class Chat:
    def __init__(self, model='llama3.3', temperature=0.2):
        self.model = model
        self.temperature = temperature
        self.client = CLIENT_OLLAMA

    async def __call__(self, content, image_path=None):
        message = {'role': 'user', 'content': content}
        if image_path is not None:
            message.update({'images': [image_path]})
        response = await self.client.chat(
            model=self.model,
            messages=[message],
            options={"temperature": self.temperature}
        )
        return response['message']['content']
    
def load_dataset_dict(folder_path, concat=False):
    dataset_name = "parquet"
    data_files = {
        "dev": f"{folder_path}/dev-0*.parquet",
        "test": f"{folder_path}/test-*.parquet",
        "validation": f"{folder_path}/validation-*.parquet"
    }
    dataset = load_dataset(dataset_name,data_files=data_files)
    if concat:
        return concatenate_datasets([dataset['dev'], dataset['test'], dataset['validation']])
    return dataset