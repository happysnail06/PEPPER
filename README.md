# PEPPER
## Prerequisites
- python == 3.9


## Installation and Usage
Follow these steps to get started:
1. **Clone the Project**:
Use Git to clone the project repository to your local machine.
   ```bash
   git clone <project git url>
   ```
2. **Download Models**
- Download fine-tuned CRS models from the [model_link](https://drive.google.com/drive/folders/1h2AcRn3cd9qXToM9hFAXFV5evXJM-wyD?usp=sharing). Please put the downloaded model in crs/utils/model directory. We follow prior research for detailed descriptions. [This Git Repository](https://github.com/RUCAIBox/iEvaLM-CRS/tree/main?tab=readme-ov-file) 
- Download "crs_data" from this [crs_data_link](https://drive.google.com/file/d/1PUx4VCI7TBNvLGIVx5h5Xnk21oUp760c/view?usp=share_link) and place the folder in the root directory.
- Download "user_data_IMDB.json" from this [user_IMDB_link](https://drive.google.com/file/d/1Jry69sSj0DVvo6haST4OzNain44xamcO/view?usp=share_link) and place the file in dataset/user_data directory.

3. **Set Openai Key** in your local environment.

4. **Generate IMDB_ReDial & IMDB_OpenDialKG** following src/data_processing.

5. **Test Run**:
   At the project's root directory, run

   ```bash
   # Install Dependencies
   pip install -r requirements.txt

   # Save item embeddings
   bash script/{dataset}/cache_item.sh

   # Generate dialogues
   bash script/{dataset}/chat_{model_name}.sh
   ```
   - The dialogues will be saved in a newly created folder: dialogue_

6. **For evaluation**:
   run

   ```bash
   python src/crs_eval.py
   ```
   - The results will be saved in a newly created folder: results_
   