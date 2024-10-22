import pandas as pd
from sklearn.metrics import classification_report

from classifier.Classifier import Classifier

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, \
    RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModel

from classifier.ErrorDataset import ErrorDataset


class BERT(Classifier):
    def __init__(self, known_errors_list: pd.DataFrame = pd.DataFrame()):
        super().__init__(known_errors_list)
        self.tokenizer = None

    def tokenize(self,
                 train_texts: list,
                 test_texts: list,
                 pretrained_model_name: str = 'microsoft/codebert-base',
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        # Tokenização
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

        return train_encodings, test_encodings

    def prepare_train(self,
                      train_dataset: pd.DataFrame,
                      test_dataset: pd.DataFrame,
                      num_labels: int,
                      output_dir: str = './results',
                      num_train_epochs: int = 50,
                      per_device_train_batch_size: int = 10,
                      per_device_eval_batch_size: int = 10,
                      warmup_steps: int = 20,
                      weight_decay: int = 0.01,
                      logging_dir: str = './logs',
                      pretrained_model_name: str = 'microsoft/codebert-base',
                      ):
        self.model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir
        )

        # Treinador
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        return trainer

    def execute(self):
        self.known_errors_list = self.load_data()
        filtered_data = self.known_errors_list[["name", "description", "classified"]]
        self.labels = filtered_data['name']

        raw_x_train, raw_x_test, raw_train_labels, raw_test_labels = self.prepare_data(
            text_dataframe=filtered_data['description'].tolist(),
            label_dataframe=filtered_data['name']
        )

        labels_dict_train = {index: valor for index, valor in enumerate(raw_train_labels)}
        labels_dict_test = {index: valor for index, valor in enumerate(raw_test_labels)}

        print(f"Train Size: {len(labels_dict_train)}")
        print(f"Test Size : {len(labels_dict_test)}")

        train_encodings, test_encodings = self.tokenize(train_texts=raw_x_train, test_texts=raw_x_test)

        train_dataset = ErrorDataset(encodings=train_encodings, labels=list(labels_dict_train.keys()))
        test_dataset = ErrorDataset(encodings=test_encodings, labels=list(labels_dict_test.keys()))

        trainer = self.prepare_train(train_dataset=train_dataset,
                                     test_dataset=test_dataset,
                                     num_labels=len(set(filtered_data['name'].tolist()))
                                     )
        trainer.train()

        predictions = trainer.predict(test_dataset)
        predicted_labels = predictions.predictions.argmax(axis=-1)

        print(f"Predicted   : {predicted_labels}")
        print(f"Test labels : {raw_test_labels.index.tolist()}")

        print(classification_report(raw_test_labels.index.tolist(),
                                    predicted_labels,
                                    zero_division=0,
                                    #target_names=self.labels.values.tolist()
                                    ))
