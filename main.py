from flask import Flask
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np

df = pd.read_csv('data/casas.csv')
# Modelo apenas com o tamanho
X = df['tamanho']
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
linear_rg = LinearRegression()
linear_rg.fit(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))

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

@app.route('/houseprices1/<tamanho>')
def previsao_one_var(tamanho):
    y_pred = linear_rg.predict([[float(tamanho)]])
    a = round(y_pred[0][0], 2)
    return str(a)


if __name__ =='__main__':
    app.run(debug=True)

