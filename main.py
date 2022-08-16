from flask import Flask
from textblob import TextBlob

app = Flask('meu_app')

@app.route('/')
def main():
    return 'Minha Primeira API'

@app.route('/sentimento/<frase>')
def sentiment(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt-bt', to='en')
    polaridade = tb_en.sentiment.polarity
    return 'A polaridade da frase é de %s' % polaridade


if __name__ =='__main__':
    app.run(debug=True)

