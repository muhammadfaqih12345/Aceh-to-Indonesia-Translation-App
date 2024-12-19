import streamlit as st
import joblib
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gtts # pip install gtts

@st.cache_resource
def load_model():
    model = joblib.load("model_T5.pkl")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    return model, tokenizer

model, tokenizer = load_model()

st.set_page_config(page_title='Translate Aceh to Indonesia', page_icon='translator-icon.png', layout='wide', initial_sidebar_state='expanded')


Languages = {'aceh':'ace','indonesia':'ind'}


translator = model
st.title("Language Translator:balloon:")

aceh_text = st.text_area("Enter text:",height=None,max_chars=None,key=None,help="Enter your text here")

option1 = st.selectbox('Input language',
                      ('aceh'))

option2 = st.selectbox('Output language',
                       ('indonesia'))

value1 = Languages[option1]
value2 = Languages[option2]

if st.button('Translate Sentence'):
    if aceh_text == "":
        st.warning('Please **enter text** for translation')

    else aceh_text.strip():
        # Prepare input
        input_text = f"translate Aceh to Indonesian: {aceh_text}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate translation
        outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display result
        st.write("### Translation:")
        st.info(str(translated_text))
        st.success("Translation is **successfully** completed!")
        st.balloons()
else:
    pass

 
 





