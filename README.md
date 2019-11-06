Hinglish NLP
====

Welcome! This repository contains NLP resources for Hinglish.

Hinglish is a compound word made from *Hindi* and *English*. Hinglish is the code mixed (and code switched) mode of communication used by the bilinguals fluent in Hindi and English. In this repository, you'll find NLP resources developed/adapted for the Hinglish data.

- Trained NLP models for Hinglish
- Effective algorithms for various tasks in Hinglish
- Data used for training
- Other Hinglish data assets


## Data Directory Structure

Here's how the data directory is structured. Some data files will not be present in the Github repo as they are not final yet or are big in size.

```
data
├── README.md
├── assets
│   ├── README.md
│   ├── eng_vocab
│   ├── hindi_chars
│   ├── stop_hindi
│   └── stop_hinglish
├── normalization
│   ├── README.md
│   ├── train_manual_all.csv
│   ├── train_released_1.csv
│   └── train_synthetic.csv
└── transliteration
    ├── README.md
    ├── en_hi_rel.csv
    └── en_hi_wiki.csv
```

The `README`s in each folder will explain in detail what each csv/txt file is and how they were created. All the citations can also be found there if the datasets were derived from other published datasets.


## Data Generation

All the data generation/creation scripts and code are present in the `datagen` directory. To execute the scripts you have to go inside the `datagen` directory.

1. To get the extracted dataset from the JSON dump of Wikidata run the scripts in the following order with all the paths and filenames changed accordingly.

    ```sh
    cd datagen
    python wikidata2.py
    python wiki_trans_align.py
    python wiki_trans_filter.py
    ```

    Note: At the end you wont get the final dataset. You'll have to process this dataset manually. More detailed process can be found at this blog entry - [Wikidata for Transliteration Pairs](https://trigonaminima.github.io/2019/11/transliteration-wikidata/)



## Blog Posts

Below are the list of blog posts I wrote in order to explain different parts of the work present in this repository and other concepts around Hinglish, Transliteration and NLP.

1. Intro to the Hinglish and Transliteration: [https://trigonaminima.github.io/2018/06/hinglish-and-transliteration/](https://trigonaminima.github.io/2018/06/hinglish-and-transliteration/);
2. What started this all: [(Mis)adventures of Building a Chat Bot](https://trigonaminima.github.io/2018/10/chatbot/);
3. Intro to WX Notation, something which I expect to use in the final model: [Understanding WX notation](https://trigonaminima.github.io/2019/03/wx_notation/);
4. Intuition of the components of the Seq2Seq models: [Seq2Seq Components](https://trigonaminima.github.io/2019/09/seq2seq-components/).
5. Training data creation for transliteration from Wikidata: [Wikidata for Transliteration Pairs](https://trigonaminima.github.io/2019/11/transliteration-wikidata/)
