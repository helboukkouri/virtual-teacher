import gradio as gr
from chatbot import generate_response
from text_to_speech import initialize_xtts_text_to_speech


input_text = gr.Textbox(label="Ask your teacher a question (English)")
language = gr.Dropdown(["fr", "es", "it", "ja"], label="Target Language")

output_text = gr.Textbox(label="Assistant Response (Text)")
output_audio = gr.Audio(type="filepath", label="Assistant Response (Audio)", autoplay=True)

convert_text_to_speech = initialize_xtts_text_to_speech()


def run_translation_with_tts(input_text, language):
    global convert_text_to_speech

    response = generate_response(
        input_text,
        language=language,
        keep_chat_history=False,
        debug=False,
    )
    convert_text_to_speech(response)#, language=language)
    return response, "output.wav"


if __name__ == "__main__":
    demo = gr.Interface(
        run_translation_with_tts,
        inputs=[input_text, language],
        outputs=[output_text, output_audio],
        title="Translation Assistant"
    )
    demo.launch()
