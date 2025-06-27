\# 🔮 VGC Restricted Core Teammate Predictor

This interactive tool predicts the most likely teammates for a given pair of restricted Pokémon in VGC Regulation I, based on battle data collected from Pokémon Showdown BO3 matches.

It uses a trained supervised machine learning model to identify common patterns and displays the results with sprite images and probabilities in a visual grid.

\---

\#\# 🚀 Run in Google Colab

\[\!\[Open In Colab\](https://colab.research.google.com/github/spidersocks/poke-team-predictor/blob/main/vgc\_teammate\_predictor\_demo.ipynb)

\---

\#\# 📁 Files Included

| File | Description |    
|------|-------------|    
| \`vgc\_teammate\_predictor\_demo.ipynb\` | The main interactive Colab notebook |    
| \`models/vgc\_model.joblib\` | Trained multi-output classifier |    
| \`data/X\_df.csv\` | Processed input features (core Pokémon) |    
| \`data/label\_columns.json\` | Ordered label list for model compatibility |

\---

\#\# 🧠 How It Works

\- Choose two restricted Pokémon from dropdown menus    
\- Click \*\*Predict Teammates\*\*    
\- The top 10 most likely partners will be shown with:    
  \- Official sprites via \[PokéAPI\](https://pokeapi.co/)    
  \- Predicted co-occurrence probabilities    
  \- Clean grid layout for comparison

\---

\#\# 📝 About the Project

This is part of our Milestone II project for the University of Michigan, focused on understanding competitive Pokémon team composition using supervised and unsupervised learning.

More details are available in our \[final report\](link-to-PDF-if-public).

\---

\#\# ✨ Example Output

\!\[Example Grid Output\](vgc\_predictor\_preview.jpg)

\---

\#\# 📜 License & Credits

\- Pokémon sprites from \[PokéAPI\](https://pokeapi.co/)    
\- Based on VGC Regulation I formats    
\- Created by Jackson Montuoro and Sean Fontaine for SIADS 696  

\*Generative AI disclaimer: generative AI was used for formatting this README. User and domain-specific information is original content.\*