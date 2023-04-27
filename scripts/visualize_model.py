#Visualise model output using streamlit:
#Allows navigating through records (next/prev/number input)
#Also allows dynamic model output by typing directly in to the left hand text box
#To customise for your own project please find sections marked ###CUSTOMISATION### and follow comments to load in your data, model, define labels and associated colours

#Import dependencies
import streamlit
import spacy
import os
import pandas as pd
import spacy_streamlit
from spacy_streamlit import visualize_spans, process_text, load_model
from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
import streamlit as st
from spacy.language import Language
from spacy import displacy
import base64

#Set streamlit page config this must come before any other streamlit references
streamlit.set_page_config(page_title=None,  layout="wide", initial_sidebar_state="expanded", menu_items=None)

@st.cache_resource
#@st.cache(allow_output_mutation=True, suppress_st_warning=True) #DEPRECATED
def load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model."""
    return spacy.load(name)

@st.cache_resource
#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def process_text(model_name: str, text: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object."""
    nlp = load_model(model_name)
    return nlp(text)

def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)

SPACY_VERSION = tuple(map(int, spacy.__version__.split(".")))


#Functions to define how to style the output dataframe---------------------------------

###CUSTOMISATION###
#Define colours for both labelling the spans in the text and for the output dataframe
#Takes a dictionary of "category":"color" pairs
colour_options = {"colors":
           {"HVL": "DeepSkyBlue",
            "NO_HVL": "LightBlue",
            "GVL": "Chartreuse",
            "NO_GVL": "LightGreen",
            "RVL": "Crimson",
            "NO_RVL": "LightCoral"}
          }

#Define a function to use as part of pandas.styles.applymap() to apply conditional formatting on the output table
def output_styles(val):
    #Check if the value is in the colour_options dictionary
    if val in colour_options["colors"]:
        #Set the output of this function to CSS making the background colour the value from the colour_options dictionary 
        styling = ('background-color: ' + str(colour_options['colors'][val]))
    #If value not in dictionary then change nothing
    else:
        styling = ''
    # Return the CSS attribute-value pair
    return styling 

#Define function to visualise spans - mostly as per spacy-streamlit source with superfluous options removed and the spancat scores added
def visualize_spans(
    doc: Union[spacy.tokens.Doc, Dict[str, str]],
    *,
    ###CUSTOMISATION### - change span key here if using something other than the default 'sc'
    spans_key: str = "sc",
    ###CUSTOMISATION### - can change the below list of attributes for the model output table
    attrs: List[str] = ["text", "label_", "start", "end", "start_char", "end_char"],
    show_table: bool = True,
    title: Optional[str] = "Spans",
    manual: bool = False,
    displacy_options: Optional[Dict] = None
):
    if SPACY_VERSION < (3, 3, 0):
        raise ValueError(
            f"'visualize_spans' requires spacy>=3.3.0. You have spacy=={spacy.__version__}"
        )
    if not displacy_options:
        displacy_options = dict()
    displacy_options["spans_key"] = spans_key

    if title:
        st.header(title)

    if manual:
        if show_table:
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'show_table' must be set to False."
            )
        if not isinstance(doc, dict):
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'doc' must be of type 'Dict', not 'spacy.tokens.Doc'."
            )
    html = displacy.render(
        doc,
        style="span",
        options=displacy_options,
        manual=manual,
    )
    st.write(f"{get_html(html)}", unsafe_allow_html=True)

    if show_table:
        data = [ [str(getattr(span, attr)) for attr in attrs] for span in doc.spans[spans_key] ]
        if data:
            df = pd.DataFrame(data, columns=attrs)
            #Add a column to the dataframe containing the spancat scores
            df["scores"] = doc.spans[spans_key].attrs["scores"]
            ###CUSTOMISATION### - change the number of decimal places here if requried
            df["scores"] = df["scores"].apply(lambda x: "{:.4f}".format(round(x, 4)))
            st.dataframe(df.style.applymap(output_styles), use_container_width=False)

###LOAD IN THE DATASET AND MODEL---------------------------------------------------------------------
###CUSTOMISATION### - Load in the dataset of text that is to be assessed, ending with a dataframe with a single column named "Text"
data_path = 'file://' + os.path.abspath('WORKING_DATA/REPORT_DATA.csv')
df = pd.read_csv(data_path)
df["Text"] = df["Scan_report_text_1"].map(str) + '. ' + df["Scan_report_text_2"].map(str)
df = df.drop(["Scan_report_text_1", "Scan_report_text_2"], axis=1)

###CUSTOMISATION### - Load in the model of your choice
#Below sets in use model as the current best trained model
spacy_model = 'training/model-best'
#use load_model from the spacy streamlit package to pre-load the model, avoids reloading model every time new text is entered which is slow
nlp = load_model(spacy_model)


###DEFINE STREAMLIT FUNCTIONS AND GENERATE FORMS------------------------------------------------

#Define the callback functions that decide what each button does
def next_case_callback():
    #increment the session state and then write it before the page updates
    streamlit.session_state.my_input = streamlit.session_state.my_input + 1
    streamlit.write(streamlit.session_state.my_input)
    
def prev_case_callback():
    #decrease the session state and then write it before the page updates
    streamlit.session_state.my_input = streamlit.session_state.my_input - 1
    streamlit.write(streamlit.session_state.my_input)
    
def form_callback():
    #write the value in the input box to the session state
    streamlit.write(streamlit.session_state.my_input)

#Generate the sidebar, and within that generate a form with the next case/prev case button and a number selector with submit button 
with streamlit.sidebar:
    with streamlit.form(key='my_form'):
        next_case = streamlit.form_submit_button(label="Next Case", on_click=next_case_callback)
        prev_case = streamlit.form_submit_button(label="Prev Case", on_click=prev_case_callback)
        number_input = streamlit.number_input(label='Index', value=0, key="my_input")
        submit_button = streamlit.form_submit_button(label='Go to row number', on_click=form_callback)

#Set up the columns to display the raw and processed text (col2 bigger than col1)       
col1, col2 = st.columns([2,3])        
        
#Generate the text input box (label collapsed so that can use a separate defined subheader to match the adjacent model output column)
with col1:
    st.subheader('Raw Report Text')
    raw_text = streamlit.text_area("Raw Report Text", df["Text"][streamlit.session_state.my_input], height=500, label_visibility="collapsed")


###PROCESS THE TEXT AND CALL VISUALIZER------------------------------------------------------------------

#Generate the Doc object from the text in the input box, also using process_text from the spacy_streamlit package
doc = process_text(spacy_model, raw_text)

#define the colours to tag spans with
options = colour_options

#Call visualise_spans on the Document object
with col2:
    st.subheader('Model Output')
    pd.options.display.float_format = '{:.2f}'.format
    visualize_spans(doc, displacy_options=options, title=None, spans_key='sc', show_table=True)
