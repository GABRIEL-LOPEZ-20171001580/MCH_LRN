<div style = 'text-align:center'>

# Tarea N°3 del curso de introducción a Machine Learning en `Python`

## Nombre del Programador: José Gabriel López Madrid
### Correo: `jglopezmad@gmail.com`

</div>



---


<div style = 'text-align:justify'>

<div style = 'text-align:center'>

## Descripción del conjunto de datos

</div>

Tenemos un conjunto de datos financieros de clientes de una institución financiera, los datos tienen relacion con las variables de _Balance_, _Pagos_, _Compras por cuotas_ ,  _Límite de Creditos_ y otros indicadores financieros de los usuarios de la institución. Nuestro objetivo es tomar esos datos y observar la tendencia entre la visualización de los datos via el algoritmo de `Analisis de Componentes Principales` y el **límite de crédito de los usuarios**, es por esto que mostramos primero los módulos utilizados para la creación de este trabajo. Se enumeran todos los módulos necesarios para correr el resto del código aquí. A continuación una pequeña tabla que describe las variables del conjunto de datos. 

<center>

| Variable               | Descripción                                                            | Unidad |
| ---------------------- | ---------------------------------------------------------------------- | ------ |
| BALANCE                | Saldo presente en la cuenta.                                           | USD    |
| PURCHASES              | Compras hechas con la tarjeta.                                         | USD    |
| ONE_OFFPURCHASES       | Compras hechas al contado.                                             | USD    |
| INSTALLMENTS_PURCHASES | Compras hechas a plazos.                                               | USD    |
| CASH_ADVANCE           | Compras realizadas pagando <br>  por adelantado.                       | USD    |
| CASH_ADVANCE_TRX       | Numero de compras realizadas  <br> via pagos por adelantado.           |        |
| CREDIT_LIMIT           | Limite de Credito de la tarjeta.                                       | USD    |
| PAYMENTS               | Pagos de la tarjeta realizados por el usuario                          | USD    |
| PRC_FULL_PAYMENT       | Porcentaje del pago total de<br>  la tarjeta realizado por el usuario. |        |

</center>


</div>


```python
# modulos de archivo
import numpy as np
import pandas as pd

#Modulos de matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.colors as clr
import matplotlib.cm as cmapp

#Modulos de sklearn para el pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer

#Modulos de sklearn para el metodo de agrupamiento jerarquico y k-medias 
from sklearn.cluster import AgglomerativeClustering, KMeans

plt.rcParams.update({
    'text.usetex':True
})
```

<div style = 'text-align:justify'>

<div style = 'text-align:center'>

## Matriz de correlacion y preprocesado de los datos

</div>
<justify>
Primero se preprocesan los datos, considerando que la data proporcionada todavía posee variables que no son de interés por el momento, sobre todo, queremos eliminar la última columna, que representa la variable `Tenure`. En este caso, no será necesario insertar dicha variable en el modelo, ya que solo tiene información de cuanto tiempo tiene el cliente con esa linea de crédito particular. Como buena práctica, siempre se hará un gráfico de la matriz de correlación para este conjunto de datos, esto con el fin de poder mostrar la matriz de la que se desprende el método de visualización que se muestra más adelante.
</justify>
</div>


```python
# Guardo los datos del .csv en una variable llamada df.  
# Por favor, usar los datos completos del dataset de Kaggle. El que se uso en clase me parece que tiene solo 1000 datos, pero queria revisar 
# Si con un mayor numero de datos, mejora el resultado. 
df = pd.read_csv('./CC_General.csv')
df = df.dropna()
#Luego obtengo las columnas que necesito para solo trabajar con los datos importantes (los que son de tipo numérico, 
#las identificaciones de los clientes no me interesan porque son variables categoricas) 
x = df[df.columns[1:11]]
#Procedo a realizar la cuestion de revisar la matriz de correlacion para revisar si los datos exhiben correlaciones 
#significativas para pasar a la reducción de dimension de los datos para su posterior visualizacion. 
mat_corr = x.corr()
mat_vals = mat_corr.values.round(3)
```


```python
#Todo este código es para poder realizar la matriz de correlacion y observar si existe correlación lineal entre los datos del conjunto. 
#Ya de entrada veo que muchas de las variables estan correlacionadas de forma débil, dado que gran parte del conjunto tiene valores que andan entre 0.4 y 0.6
#de correlación, que no es malo, pero es débil. 
#Aqui coloco los parametros para realizar una imagen esteticamente satisfactoria. 
plt.style.use('default')
pltlabels = ['$BA$','$PRC$','$OF_{PRC}$','$IS_{PRC}$'
             ,'$CA$','$TRX_{CA}$','$TRX_{PRC}$','$C_{LIM}$',
             '$PAY$','$FP_{PRC}$'] # Las etiquetas para los cuadros

#Creacion del esquema de colores, usaré un gradiente con tres colores para realizar el grafico. 

clist = [
    (246/255, 193/255, 119/255),
    (1,1,1),
    (49/255, 116/255, 143/255)
]

cm = clr.LinearSegmentedColormap.from_list(name = 'rose_pine', colors = clist, N=128)

plt.figure(figsize=(10,8)) # Tamaño de figura
plt.imshow(X = mat_corr,cmap = cm) # Crear el plot, con la imagen


for i in range(0,10):
    for j in range(0,10):
        plt.text(i,j,mat_vals[i][j], va = 'center', ha = 'center') # Colocar los números que tiene cada coeficiente de correlacion


plt.colorbar().ax.set_ylabel('Coeficiente de correlacion') # Barra de color y titulo de eje
plt.xticks(range(0,10),labels=pltlabels, rotation = 'vertical') #Colocar las etiquetas del eje x
plt.yticks(range(0,10),labels=pltlabels) #Colocar las etiquetas del eje y
plt.title("Matriz de correlacion para el conjunto de datos de tarjetas de crédito") # Titulo
plt.show() #Buena práctica para Python.
```


    
![png](output_5_0.png)
    


## Escalado de datos y preprocesado

Se procede a escalar los datos, por temas de visualizacion, queria observar si dependiendo de la tecnica de visualizacion cambia el resultado final, por esto vamos a realizar dos escalados, que son los que siguen:

1.   Escalado utilizando el `StandardScaler()`, que intenta generar una media de cero y una desviacion estandar de 1 en la escala de la data disponible, el problema de este escalado es que es sensible a valores atipicos.

2.   Escalado usando la funcion `RobustScaler()`, que busca realizar lo mismo que el escalado anterior, pero utiliza otros metodos para realizar dicho escalado (basado en cuantiles), lo que facilita un escalado mas robusto frente a valores atipicos, luego lo que se realiza es utilizar una transformacion de tipo no lineal, para ajustar los valores a que tengan una media de 0 y una desviacion estandar de 1.

Se puede observar que existe una mejora entre los dos metodos, sobre todo, el escalado estandar es, hasta cierto punto, mucho mas dificil de discernir, ademas de tener una cantidad grande de valores anomalos, que podrian cambiar el resultado de una forma significativa. Mientras que el escalado robusto tiene mejores capacidades para la visualizacion de los resultados, lo que lo hace mas idoneo para la presentacion de los resultados que fueron determinados en este cuaderno. 




```python
# Se observa que hay una correlacion debil con la cuestion de ciertas variables.

# Se procede a realizar el algoritmo de PCA. 

#x_tr = QuantileTransformer().fit_transform(X=x)
x_tr = RobustScaler().fit_transform(X=x)# Se realiza el escalado de los datos.
x_tr = PowerTransformer().fit_transform(X=x_tr)
x_std = StandardScaler().fit_transform(X=x)


pca1 = PCA(n_components=2)
pca2 = PCA(n_components=2)
vis_x = pca1.fit_transform(x_std)
vis_x2 = pca2.fit_transform(x_tr)
```


```python
# Se procede a realizar la visualizacion de la data utilizando PCA

plt.style.use('https://raw.githubusercontent.com/h4pZ/rose-pine-matplotlib/main/themes/rose-pine.mplstyle')
plt.figure(figsize=(12,10))

plt.subplot(1,2,1)
plt.grid()
plt.scatter(vis_x[:,0],vis_x[:,1])
plt.xlabel('Componente principal $C_1$')
plt.ylabel('Componente principal $C_2$')
plt.title('Escalado estandar')

plt.subplot(1,2,2)
plt.grid()
plt.scatter(vis_x2[:,0],vis_x2[:,1])
plt.xlabel('Componente principal $C_1$')
plt.ylabel('Componente principal $C_2$')
plt.title('Escalado no lineal')

plt.suptitle('Visualización de los datos usando el algoritmo de PCA')
plt.tight_layout()
plt.show()
#Con el escalado estandar pareciera que hay alrededor de tres puntos que son outliers y están relativamente lejos de los clusters que quiero ver. 
#Comparamos este resultado con resultado del escalado por cuantiles. 
```


    
![png](output_8_0.png)
    


Se utiliza el metodo del codo para poder determinar un numero optimo para el k usado en el metodo de k-medias. Se muestran a continuacion los resultados para ambos escalados, es notorio ver que el codo pareciera ser mas pronunciado con el escalado robusto, por lo que pareciera que utilizar este escalado mejora los resultados de los metodos no supervisados. 


```python
#Parametros para inicializar el metodo del codo, para obtener un k optimo para lo que sería el algoritmo de clustering,
#aunque no puedo dilucidar el codo con detalle, por lo que voy a intentar utilizar otro método, investigando,
#me encontré que había un segundo método para observar esto, que sería usando el coeficiente de silueta. 
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

plt.figure(figsize=(12,8))
for i in [1,2]:
    plt.subplot(1,2,i)
    sse = []
    m = 20
    if i == 1:
        a = x_std
    else:
        a = x_tr
    for k in range(1, m):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(a)
        sse.append(kmeans.inertia_)

    etiquetas = ['Lineal', 'Gaussiano']
    plt.grid()
    plt.plot(range(1, m), sse, marker='o')
    plt.xticks(range(1, m))
    plt.xlabel("Numero de agrupamientos")
    plt.ylabel("$\sum \\varepsilon_i ^2$",loc='top',  rotation = 'horizontal')
    plt.title(f'Escalado {etiquetas[i-1]}')
plt.tight_layout()
plt.suptitle('Metodo del codo para los dos escalados')
plt.subplots_adjust(top=0.85)
plt.show()

```


    
![png](output_10_0.png)
    


## Modelo de agrupamiento jerarquico 
Se genera el modelo de agrupamiento jerarquico en las siguientes celdas de codigo, lo notorio es que mas adelante se muestra que el metodo de agrupamiento jerarquico entrega resultados parecidos para ambos escalados, lo que da a entender que la ventaja de un escalado por sobre otro es en la visualizacion, nada mas. Si se utiliza el mismo modelo los resultados son los mismos, mientras que si se utiliza un modelo diferente, como el mostrado a continuacion, los resultados difieren dado que el escalado estandar es mas sensible a valores atipicos. Es por esto que los grupos con el limite de credito mas alto solo son conformados por pocos individuos, con valores altos de limite de credito. Es notorio observar que los resultados del segundo modelo difieren por el preprocesado, dado que se elimino la influencia de los valores anomalos, se observa una clara tendencia a tener grupos mas homogeneos


```python
# Numero de clusters para realizar la prediccion, parece ser que nuestro valor objetivo es
# cercano a 5 según el método del codo, probaré primero con 5 y luego con los vecinos más cercanos. 
n_cluster1 = 4
n_cluster2 = 4

jerarq = AgglomerativeClustering(n_clusters=n_cluster1)
jerarq = jerarq.fit(x_std)
jerarq2 = AgglomerativeClustering(n_clusters=n_cluster2)
jerarq2 = jerarq2.fit(x_tr)
```


```python
#obtenemos las categorias predichas 
predictions = jerarq.labels_

#realizamos la visualizacion 

dict_col = {
    0:'#eb6f92',
    1:'#f6c177',
    2:'#9ccfd8',
    3:'#3e8fb0',
    4:'#c4a7e7',
    5:'#e0def4'
}

leyendas = [
    "$G_0$",
    '$G_1$',
    '$G_2$',
    '$G_3$',
    '$G_5$'
]

cred_lim1 = [np.mean(x["CREDIT_LIMIT"][predictions == i]) for i in range(0,n_cluster1)]

a = sorted(range(len(cred_lim1)), key=lambda k: cred_lim1[k])

# Visualizamos la prediccion para observar los resultados del algoritmo de clasificación, comenzando con el escalado estandar
 
plt.figure(figsize=(13,8))

#Visualizacion de la prediccion usando el escalado de datos de tipo estandar
plt.subplot(1,2,1)
plt.grid()
for i in range(0,n_cluster1):
    plt.scatter(vis_x[:,0][predictions == i],vis_x[:,1][predictions == i], c=dict_col[i])
plt.xlabel('Componente principal $C_1$')
plt.ylabel('Componente principal $C_2$')
plt.legend(leyendas)
plt.title('Visualizacion resultante')

plt.subplot(1,2,2)
for i in range(0,len(np.unique(predictions))):
    plt.barh(leyendas[i],cred_lim1[i],color = dict_col[i])
    plt.text(cred_lim1[i]/2,i,f"{cred_lim1[i]:.02f} USD",ha='center',va='center',color='#26233a')
plt.xlabel('Dolares (USD)')
plt.title('Limite de Credito')



plt.suptitle('Segmentación de los clientes usando el método de agrupamiento jerárquico (escalado estandar)')
plt.tight_layout()
plt.show()

```


    
![png](output_13_0.png)
    



```python
m_list = []
tab = pd.DataFrame()
for i in range(0,np.shape(x)[1]):
    for j in a:
        m_list.append(np.mean(x[x.columns[i]][predictions == j]))
    tab[x.columns[i]] = m_list 
    m_list = []

tab = tab.rename(index={0:'Grupo A', 1:'Grupo B', 2:'Grupo C', 3: 'Grupo D'})
tab = tab.transpose()       
tab.style
```




<style type="text/css">
</style>
<table id="T_2189b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2189b_level0_col0" class="col_heading level0 col0" >Grupo A</th>
      <th id="T_2189b_level0_col1" class="col_heading level0 col1" >Grupo B</th>
      <th id="T_2189b_level0_col2" class="col_heading level0 col2" >Grupo C</th>
      <th id="T_2189b_level0_col3" class="col_heading level0 col3" >Grupo D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2189b_level0_row0" class="row_heading level0 row0" >BALANCE</th>
      <td id="T_2189b_row0_col0" class="data row0 col0" >925.870705</td>
      <td id="T_2189b_row0_col1" class="data row0 col1" >4433.285708</td>
      <td id="T_2189b_row0_col2" class="data row0 col2" >3269.676344</td>
      <td id="T_2189b_row0_col3" class="data row0 col3" >5706.937574</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row1" class="row_heading level0 row1" >PURCHASES</th>
      <td id="T_2189b_row1_col0" class="data row1 col0" >634.664532</td>
      <td id="T_2189b_row1_col1" class="data row1 col1" >569.090000</td>
      <td id="T_2189b_row1_col2" class="data row1 col2" >5079.533005</td>
      <td id="T_2189b_row1_col3" class="data row1 col3" >25277.003793</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row2" class="row_heading level0 row2" >ONEOFF_PURCHASES</th>
      <td id="T_2189b_row2_col0" class="data row2 col0" >329.493409</td>
      <td id="T_2189b_row2_col1" class="data row2 col1" >359.637529</td>
      <td id="T_2189b_row2_col2" class="data row2 col2" >3325.110017</td>
      <td id="T_2189b_row2_col3" class="data row2 col3" >18378.876207</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row3" class="row_heading level0 row3" >INSTALLMENTS_PURCHASES</th>
      <td id="T_2189b_row3_col0" class="data row3 col0" >305.444392</td>
      <td id="T_2189b_row3_col1" class="data row3 col1" >209.527218</td>
      <td id="T_2189b_row3_col2" class="data row3 col2" >1755.506461</td>
      <td id="T_2189b_row3_col3" class="data row3 col3" >6898.127586</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row4" class="row_heading level0 row4" >CASH_ADVANCE</th>
      <td id="T_2189b_row4_col0" class="data row4 col0" >405.332434</td>
      <td id="T_2189b_row4_col1" class="data row4 col1" >4647.371073</td>
      <td id="T_2189b_row4_col2" class="data row4 col2" >474.369114</td>
      <td id="T_2189b_row4_col3" class="data row4 col3" >1922.942695</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row5" class="row_heading level0 row5" >CASH_ADVANCE_TRX</th>
      <td id="T_2189b_row5_col0" class="data row5 col0" >1.636517</td>
      <td id="T_2189b_row5_col1" class="data row5 col1" >13.796639</td>
      <td id="T_2189b_row5_col2" class="data row5 col2" >1.465776</td>
      <td id="T_2189b_row5_col3" class="data row5 col3" >3.793103</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row6" class="row_heading level0 row6" >PURCHASES_TRX</th>
      <td id="T_2189b_row6_col0" class="data row6 col0" >10.584911</td>
      <td id="T_2189b_row6_col1" class="data row6 col1" >8.402521</td>
      <td id="T_2189b_row6_col2" class="data row6 col2" >69.979967</td>
      <td id="T_2189b_row6_col3" class="data row6 col3" >146.724138</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row7" class="row_heading level0 row7" >CREDIT_LIMIT</th>
      <td id="T_2189b_row7_col0" class="data row7 col0" >3573.457241</td>
      <td id="T_2189b_row7_col1" class="data row7 col1" >7822.608862</td>
      <td id="T_2189b_row7_col2" class="data row7 col2" >8289.649416</td>
      <td id="T_2189b_row7_col3" class="data row7 col3" >16003.448276</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row8" class="row_heading level0 row8" >PAYMENTS</th>
      <td id="T_2189b_row8_col0" class="data row8 col0" >1047.462208</td>
      <td id="T_2189b_row8_col1" class="data row8 col1" >3917.875662</td>
      <td id="T_2189b_row8_col2" class="data row8 col2" >4405.570309</td>
      <td id="T_2189b_row8_col3" class="data row8 col3" >25550.578535</td>
    </tr>
    <tr>
      <th id="T_2189b_level0_row9" class="row_heading level0 row9" >PRC_FULL_PAYMENT</th>
      <td id="T_2189b_row9_col0" class="data row9 col0" >0.169364</td>
      <td id="T_2189b_row9_col1" class="data row9 col1" >0.049923</td>
      <td id="T_2189b_row9_col2" class="data row9 col2" >0.158878</td>
      <td id="T_2189b_row9_col3" class="data row9 col3" >0.463297</td>
    </tr>
  </tbody>
</table>




<div style = 'text-align:justify'>

Como se observa aca, el resultado de la tranformacion con la funcion de Yeo-Johnson realizo un escalado mas afin a una distribucion estandar. Por lo que los grupos se encuentran mas mezclados, con esto se explica que se tenga una variacion menos pronunciada con respecto a los diferentes grupos de estudio.

</div>


```python
#Observemos el limite de credito de cada grupo, es interesante notar que dependiendo de los 
#parametros y el escalado, se pueden notar tendencias diferentes, el escalado con potencias
#o gaussiano, pareciera que tiene la bondad de limpiar valores atipicos, aunque aumenta el
#mezclado de ciertas categorias, por lo que no es de extrañar que muchos de los resultados finales difieran de los del apartado anterior. 
predictions2 = jerarq2.labels_
#list comprehension para generar un vector con los promedios de credito para cada grupo
cred_lim2 = [np.mean(x["CREDIT_LIMIT"][predictions2 == i]) for i in range(0,n_cluster2)]
b = sorted(range(len(cred_lim2)), key=lambda k: cred_lim2[k])
#Voy a realizar un grafico de barras para comparar los resultados de ambos escalados. 
plt.figure(figsize=(13,8))
plt.subplot(1,2,1)
plt.grid()
for i in range(0,n_cluster2):
    plt.scatter(vis_x2[:,0][predictions2 == i],vis_x2[:,1][predictions2 == i], c=dict_col[i])
plt.xlabel('Componente principal $C_1$')
plt.ylabel('Componente principal $C_2$')
plt.legend(leyendas)
plt.title('Visualizacion resultante')

plt.subplot(1,2,2)
for i in range(0,len(np.unique(predictions))):
    plt.barh(leyendas[i],cred_lim2[i],color = dict_col[i])
    plt.text(cred_lim2[i]/2,i,f"{cred_lim2[i]:.02f} USD",ha='center',va='center',color='#26233a')
plt.xlabel('Dolares (USD)')
plt.title('Limite de Credito')

plt.suptitle('Segmentación de los clientes usando el método de agrupamiento jerárquico (escalado gaussiano)')
plt.tight_layout()
plt.show()

```


    
![png](output_16_0.png)
    



```python
m_list = []
tab = pd.DataFrame()
for i in range(0,np.shape(x)[1]):
    for j in b:
        m_list.append(np.mean(x[x.columns[i]][predictions2 == j]))
    tab[x.columns[i]] = m_list 
    m_list = []

tab = tab.rename(index={0:'Grupo A', 1:'Grupo B', 2:'Grupo C', 3: 'Grupo D'})
tab = tab.transpose()       
tab.style
```




<style type="text/css">
</style>
<table id="T_5ba6f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_5ba6f_level0_col0" class="col_heading level0 col0" >Grupo A</th>
      <th id="T_5ba6f_level0_col1" class="col_heading level0 col1" >Grupo B</th>
      <th id="T_5ba6f_level0_col2" class="col_heading level0 col2" >Grupo C</th>
      <th id="T_5ba6f_level0_col3" class="col_heading level0 col3" >Grupo D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5ba6f_level0_row0" class="row_heading level0 row0" >BALANCE</th>
      <td id="T_5ba6f_row0_col0" class="data row0 col0" >625.928154</td>
      <td id="T_5ba6f_row0_col1" class="data row0 col1" >2318.702887</td>
      <td id="T_5ba6f_row0_col2" class="data row0 col2" >2728.037759</td>
      <td id="T_5ba6f_row0_col3" class="data row0 col3" >1483.765869</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row1" class="row_heading level0 row1" >PURCHASES</th>
      <td id="T_5ba6f_row1_col0" class="data row1 col0" >654.269730</td>
      <td id="T_5ba6f_row1_col1" class="data row1 col1" >47.209627</td>
      <td id="T_5ba6f_row1_col2" class="data row1 col2" >1531.355574</td>
      <td id="T_5ba6f_row1_col3" class="data row1 col3" >3600.514799</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row2" class="row_heading level0 row2" >ONEOFF_PURCHASES</th>
      <td id="T_5ba6f_row2_col0" class="data row2 col0" >322.429090</td>
      <td id="T_5ba6f_row2_col1" class="data row2 col1" >43.075831</td>
      <td id="T_5ba6f_row2_col2" class="data row2 col2" >898.583716</td>
      <td id="T_5ba6f_row2_col3" class="data row2 col3" >2311.189353</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row3" class="row_heading level0 row3" >INSTALLMENTS_PURCHASES</th>
      <td id="T_5ba6f_row3_col0" class="data row3 col0" >332.434278</td>
      <td id="T_5ba6f_row3_col1" class="data row3 col1" >4.181603</td>
      <td id="T_5ba6f_row3_col2" class="data row3 col2" >632.988757</td>
      <td id="T_5ba6f_row3_col3" class="data row3 col3" >1289.325446</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row4" class="row_heading level0 row4" >CASH_ADVANCE</th>
      <td id="T_5ba6f_row4_col0" class="data row4 col0" >13.416801</td>
      <td id="T_5ba6f_row4_col1" class="data row4 col1" >2126.478184</td>
      <td id="T_5ba6f_row4_col2" class="data row4 col2" >2215.756787</td>
      <td id="T_5ba6f_row4_col3" class="data row4 col3" >16.422266</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row5" class="row_heading level0 row5" >CASH_ADVANCE_TRX</th>
      <td id="T_5ba6f_row5_col0" class="data row5 col0" >0.091270</td>
      <td id="T_5ba6f_row5_col1" class="data row5 col1" >6.880550</td>
      <td id="T_5ba6f_row5_col2" class="data row5 col2" >7.535811</td>
      <td id="T_5ba6f_row5_col3" class="data row5 col3" >0.058566</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row6" class="row_heading level0 row6" >PURCHASES_TRX</th>
      <td id="T_5ba6f_row6_col0" class="data row6 col0" >10.932804</td>
      <td id="T_5ba6f_row6_col1" class="data row6 col1" >0.662868</td>
      <td id="T_5ba6f_row6_col2" class="data row6 col2" >24.751351</td>
      <td id="T_5ba6f_row6_col3" class="data row6 col3" >45.461538</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row7" class="row_heading level0 row7" >CREDIT_LIMIT</th>
      <td id="T_5ba6f_row7_col0" class="data row7 col0" >3614.826452</td>
      <td id="T_5ba6f_row7_col1" class="data row7 col1" >4231.147824</td>
      <td id="T_5ba6f_row7_col2" class="data row7 col2" >5183.129129</td>
      <td id="T_5ba6f_row7_col3" class="data row7 col3" >7095.700890</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row8" class="row_heading level0 row8" >PAYMENTS</th>
      <td id="T_5ba6f_row8_col0" class="data row8 col0" >882.049609</td>
      <td id="T_5ba6f_row8_col1" class="data row8 col1" >1714.031574</td>
      <td id="T_5ba6f_row8_col2" class="data row8 col2" >2700.757556</td>
      <td id="T_5ba6f_row8_col3" class="data row8 col3" >3337.543161</td>
    </tr>
    <tr>
      <th id="T_5ba6f_level0_row9" class="row_heading level0 row9" >PRC_FULL_PAYMENT</th>
      <td id="T_5ba6f_row9_col0" class="data row9 col0" >0.208544</td>
      <td id="T_5ba6f_row9_col1" class="data row9 col1" >0.040242</td>
      <td id="T_5ba6f_row9_col2" class="data row9 col2" >0.060538</td>
      <td id="T_5ba6f_row9_col3" class="data row9 col3" >0.345664</td>
    </tr>
  </tbody>
</table>




## Metodo de k-medias

<div style = 'text-align:justify'>

Se implementa el modelo de k-medias en las siguientes lineas de código, además de esto se muestra la respectiva visualización más adelante.

</div>


```python
#Metodo de k-medias
k = 4

k_medias1 = KMeans( n_clusters = k, n_init=10)
k_medias1 = k_medias1.fit( x_std )
k_medias2 = KMeans( n_clusters = k, n_init=10)
k_medias2 = k_medias2.fit( x_tr )
```


```python
#obtenemos las categorias predichas 
predictions = k_medias1.labels_

#realizamos la visualizacion 

dict_col = {
    0:'#eb6f92',
    1:'#f6c177',
    2:'#9ccfd8',
    3:'#3e8fb0',
    4:'#c4a7e7',
    5:'#e0def4'
}

leyendas = [
    "$G_0$",
    '$G_1$',
    '$G_2$',
    '$G_3$',
    '$G_5$'
]

cred_lim1 = [np.mean(x["CREDIT_LIMIT"][predictions == i]) for i in range(0,n_cluster1)]
a = sorted(range(len(cred_lim1)), key=lambda k: cred_lim1[k])

# Visualizamos la prediccion para observar los resultados del algoritmo de clasificación, comenzando con el escalado estandar
 
plt.figure(figsize=(13,9))

#Visualizacion de la prediccion usando el escalado de datos de tipo estandar
plt.subplot(1,2,1)
plt.grid()
for i in range(0,n_cluster1):
    plt.scatter(vis_x[:,0][predictions == i],vis_x[:,1][predictions == i], c=dict_col[i])
plt.xlabel('Componente principal $C_1$')
plt.ylabel('Componente principal $C_2$')
plt.legend(leyendas)
plt.title('visualizacion resultante')

plt.subplot(1,2,2)
for j in range(0,len(np.unique(predictions))):
    plt.barh(leyendas[j],cred_lim1[j],color = dict_col[j])
    plt.text(x=cred_lim1[j]/2 ,y=j,s=f"{cred_lim1[j]:.02f} USD",ha='center',va='center',color='#26233a')
plt.xlabel("Dolares (USD)")
plt.title('Limite de crédito')



plt.suptitle('Segmentación de los clientes usando el método de k-medias (escalado estandar)')
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    



```python
m_list = []
tab = pd.DataFrame()
for i in range(0,np.shape(x)[1]):
    for j in a:
        m_list.append(np.mean(x[x.columns[i]][predictions == j]))
    tab[x.columns[i]] = m_list 
    m_list = []

tab = tab.rename(index={0:'Grupo A', 1:'Grupo B', 2:'Grupo C', 3: 'Grupo D'})
tab = tab.transpose()       
tab.style
```




<style type="text/css">
</style>
<table id="T_6c518">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6c518_level0_col0" class="col_heading level0 col0" >Grupo A</th>
      <th id="T_6c518_level0_col1" class="col_heading level0 col1" >Grupo B</th>
      <th id="T_6c518_level0_col2" class="col_heading level0 col2" >Grupo C</th>
      <th id="T_6c518_level0_col3" class="col_heading level0 col3" >Grupo D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6c518_level0_row0" class="row_heading level0 row0" >BALANCE</th>
      <td id="T_6c518_row0_col0" class="data row0 col0" >1132.278034</td>
      <td id="T_6c518_row0_col1" class="data row0 col1" >140.256067</td>
      <td id="T_6c518_row0_col2" class="data row0 col2" >4947.140037</td>
      <td id="T_6c518_row0_col3" class="data row0 col3" >3661.249613</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row1" class="row_heading level0 row1" >PURCHASES</th>
      <td id="T_6c518_row1_col0" class="data row1 col0" >578.500854</td>
      <td id="T_6c518_row1_col1" class="data row1 col1" >1259.504454</td>
      <td id="T_6c518_row1_col2" class="data row1 col2" >638.934080</td>
      <td id="T_6c518_row1_col3" class="data row1 col3" >7868.751305</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row2" class="row_heading level0 row2" >ONEOFF_PURCHASES</th>
      <td id="T_6c518_row2_col0" class="data row2 col0" >336.019156</td>
      <td id="T_6c518_row2_col1" class="data row2 col1" >614.645192</td>
      <td id="T_6c518_row2_col2" class="data row2 col2" >392.268018</td>
      <td id="T_6c518_row2_col3" class="data row2 col3" >5158.318146</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row3" class="row_heading level0 row3" >INSTALLMENTS_PURCHASES</th>
      <td id="T_6c518_row3_col0" class="data row3 col0" >242.785835</td>
      <td id="T_6c518_row3_col1" class="data row3 col1" >644.945293</td>
      <td id="T_6c518_row3_col2" class="data row3 col2" >246.776482</td>
      <td id="T_6c518_row3_col3" class="data row3 col3" >2711.999739</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row4" class="row_heading level0 row4" >CASH_ADVANCE</th>
      <td id="T_6c518_row4_col0" class="data row4 col0" >507.111981</td>
      <td id="T_6c518_row4_col1" class="data row4 col1" >91.386682</td>
      <td id="T_6c518_row4_col2" class="data row4 col2" >4754.738514</td>
      <td id="T_6c518_row4_col3" class="data row4 col3" >611.996726</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row5" class="row_heading level0 row5" >CASH_ADVANCE_TRX</th>
      <td id="T_6c518_row5_col0" class="data row5 col0" >2.079664</td>
      <td id="T_6c518_row5_col1" class="data row5 col1" >0.320318</td>
      <td id="T_6c518_row5_col2" class="data row5 col2" >13.628571</td>
      <td id="T_6c518_row5_col3" class="data row5 col3" >1.984334</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row6" class="row_heading level0 row6" >PURCHASES_TRX</th>
      <td id="T_6c518_row6_col0" class="data row6 col0" >9.795811</td>
      <td id="T_6c518_row6_col1" class="data row6 col1" >18.819957</td>
      <td id="T_6c518_row6_col2" class="data row6 col2" >9.658929</td>
      <td id="T_6c518_row6_col3" class="data row6 col3" >92.467363</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row7" class="row_heading level0 row7" >CREDIT_LIMIT</th>
      <td id="T_6c518_row7_col0" class="data row7 col0" >3382.206945</td>
      <td id="T_6c518_row7_col1" class="data row7 col1" >4808.895791</td>
      <td id="T_6c518_row7_col2" class="data row7 col2" >8393.084416</td>
      <td id="T_6c518_row7_col3" class="data row7 col3" >9565.404700</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row8" class="row_heading level0 row8" >PAYMENTS</th>
      <td id="T_6c518_row8_col0" class="data row8 col0" >1007.707523</td>
      <td id="T_6c518_row8_col1" class="data row8 col1" >1548.109421</td>
      <td id="T_6c518_row8_col2" class="data row8 col2" >3936.509350</td>
      <td id="T_6c518_row8_col3" class="data row8 col3" >7446.401885</td>
    </tr>
    <tr>
      <th id="T_6c518_level0_row9" class="row_heading level0 row9" >PRC_FULL_PAYMENT</th>
      <td id="T_6c518_row9_col0" class="data row9 col0" >0.035287</td>
      <td id="T_6c518_row9_col1" class="data row9 col1" >0.746153</td>
      <td id="T_6c518_row9_col2" class="data row9 col2" >0.035651</td>
      <td id="T_6c518_row9_col3" class="data row9 col3" >0.234838</td>
    </tr>
  </tbody>
</table>





```python
#Observemos el limite de credito de cada grupo, es interesante notar que dependiendo de los parametros
#y el escalado, se pueden notar tendencias diferentes, el escalado con potencias o gaussiano,
#pareciera que tiene la bondad de limpiar valores atipicos, aunque aumenta el mezclado de ciertas categorias
#, por lo que no es de extrañar que muchos de los resultados finales difieran de los del apartado anterior. 
predictions2 = k_medias2.labels_
#list comprehension para generar un vector con los promedios de credito para cada grupo
cred_lim2 = [np.mean(x["CREDIT_LIMIT"][predictions2 == i]) for i in range(0,n_cluster2)]
b = sorted(range(len(cred_lim2)), key=lambda k: cred_lim2[k])
#Voy a realizar un grafico de barras para comparar los resultados de ambos escalados. 
plt.figure(figsize=(13,9))
plt.subplot(1,2,1)
plt.grid()
for i in range(0,n_cluster2):
    plt.scatter(vis_x2[:,0][predictions2 == i],vis_x2[:,1][predictions2 == i], c=dict_col[i])
plt.xlabel('Componente principal $C_1$')
plt.ylabel('Componente principal $C_2$')
plt.legend(leyendas)
plt.title('Resultado con escalado gaussiano')

plt.subplot(1,2,2)
for j in range(0,len(np.unique(predictions))):
    plt.barh(leyendas[j],cred_lim2[j],color = dict_col[j])
    plt.text(x = cred_lim2[j]/2,y = j,s = f"{cred_lim2[j]:.02f} USD",ha='center',va='center',color='black')
plt.xlabel('Dolares (USD)')
plt.title('Limite de credito (escalado gaussiano)')

plt.suptitle('Segmentación de los clientes usando el método de k-medias (escalado gaussiano)')
plt.tight_layout()
plt.show()
```


    
![png](output_22_0.png)
    



```python
m_list = []
tab = pd.DataFrame()
for i in range(0,np.shape(x)[1]):
    for j in b:
        m_list.append(np.mean(x[x.columns[i]][predictions2 == j]))
    tab[x.columns[i]] = m_list 
    m_list = []

tab = tab.rename(index={0:'Grupo A', 1:'Grupo B', 2:'Grupo C', 3: 'Grupo D'})
tab = tab.transpose()       
tab.style
```




<style type="text/css">
</style>
<table id="T_f0b10">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f0b10_level0_col0" class="col_heading level0 col0" >Grupo A</th>
      <th id="T_f0b10_level0_col1" class="col_heading level0 col1" >Grupo B</th>
      <th id="T_f0b10_level0_col2" class="col_heading level0 col2" >Grupo C</th>
      <th id="T_f0b10_level0_col3" class="col_heading level0 col3" >Grupo D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f0b10_level0_row0" class="row_heading level0 row0" >BALANCE</th>
      <td id="T_f0b10_row0_col0" class="data row0 col0" >417.144355</td>
      <td id="T_f0b10_row0_col1" class="data row0 col1" >2258.759775</td>
      <td id="T_f0b10_row0_col2" class="data row0 col2" >1321.178164</td>
      <td id="T_f0b10_row0_col3" class="data row0 col3" >3593.073402</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row1" class="row_heading level0 row1" >PURCHASES</th>
      <td id="T_f0b10_row1_col0" class="data row1 col0" >366.226344</td>
      <td id="T_f0b10_row1_col1" class="data row1 col1" >52.654884</td>
      <td id="T_f0b10_row1_col2" class="data row1 col2" >2777.969729</td>
      <td id="T_f0b10_row1_col3" class="data row1 col3" >1744.359964</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row2" class="row_heading level0 row2" >ONEOFF_PURCHASES</th>
      <td id="T_f0b10_row2_col0" class="data row2 col0" >138.538836</td>
      <td id="T_f0b10_row2_col1" class="data row2 col1" >38.233017</td>
      <td id="T_f0b10_row2_col2" class="data row2 col2" >1730.703476</td>
      <td id="T_f0b10_row2_col3" class="data row2 col3" >1064.871174</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row3" class="row_heading level0 row3" >INSTALLMENTS_PURCHASES</th>
      <td id="T_f0b10_row3_col0" class="data row3 col0" >228.282496</td>
      <td id="T_f0b10_row3_col1" class="data row3 col1" >14.460598</td>
      <td id="T_f0b10_row3_col2" class="data row3 col2" >1047.603752</td>
      <td id="T_f0b10_row3_col3" class="data row3 col3" >679.508559</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row4" class="row_heading level0 row4" >CASH_ADVANCE</th>
      <td id="T_f0b10_row4_col0" class="data row4 col0" >30.868615</td>
      <td id="T_f0b10_row4_col1" class="data row4 col1" >2054.689360</td>
      <td id="T_f0b10_row4_col2" class="data row4 col2" >23.224124</td>
      <td id="T_f0b10_row4_col3" class="data row4 col3" >2866.907315</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row5" class="row_heading level0 row5" >CASH_ADVANCE_TRX</th>
      <td id="T_f0b10_row5_col0" class="data row5 col0" >0.211030</td>
      <td id="T_f0b10_row5_col1" class="data row5 col1" >6.770667</td>
      <td id="T_f0b10_row5_col2" class="data row5 col2" >0.138148</td>
      <td id="T_f0b10_row5_col3" class="data row5 col3" >9.213523</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row6" class="row_heading level0 row6" >PURCHASES_TRX</th>
      <td id="T_f0b10_row6_col0" class="data row6 col0" >8.347466</td>
      <td id="T_f0b10_row6_col1" class="data row6 col1" >0.960762</td>
      <td id="T_f0b10_row6_col2" class="data row6 col2" >35.324770</td>
      <td id="T_f0b10_row6_col3" class="data row6 col3" >26.752669</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row7" class="row_heading level0 row7" >CREDIT_LIMIT</th>
      <td id="T_f0b10_row7_col0" class="data row7 col0" >2862.855556</td>
      <td id="T_f0b10_row7_col1" class="data row7 col1" >4158.427128</td>
      <td id="T_f0b10_row7_col2" class="data row7 col2" >6341.022317</td>
      <td id="T_f0b10_row7_col3" class="data row7 col3" >6443.638790</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row8" class="row_heading level0 row8" >PAYMENTS</th>
      <td id="T_f0b10_row8_col0" class="data row8 col0" >530.229570</td>
      <td id="T_f0b10_row8_col1" class="data row8 col1" >1669.631697</td>
      <td id="T_f0b10_row8_col2" class="data row8 col2" >2773.064997</td>
      <td id="T_f0b10_row8_col3" class="data row8 col3" >3331.567598</td>
    </tr>
    <tr>
      <th id="T_f0b10_level0_row9" class="row_heading level0 row9" >PRC_FULL_PAYMENT</th>
      <td id="T_f0b10_row9_col0" class="data row9 col0" >0.196985</td>
      <td id="T_f0b10_row9_col1" class="data row9 col1" >0.034892</td>
      <td id="T_f0b10_row9_col2" class="data row9 col2" >0.299186</td>
      <td id="T_f0b10_row9_col3" class="data row9 col3" >0.043586</td>
    </tr>
  </tbody>
</table>





```python

```

<div style = 'text-align:justify'>

## Interpretación del método de k-medias

Puede notarse que los resultados entre ambos tipos de escalado presentan una tendencia similar: dos grupos con un mayor limite de credito en sus tarjetas y dos grupos con un limite menor en sus tarjetas, con una diferencia de más de 2000 USD que los separa. De nuevo, el escalado puede jugar en contra, en cuanto a que el escalado realizado con `PowerTransformer` tiene un efecto de reducción de la variabilidad de los datos procesados, por lo que el efecto de valores atípicos se reduce significativamente, por lo que, para concluir, este segundo método pareciera que podría usarse para indicar una cota inferior de la que poder partir para segmentar a los clientes de la institución.


</div>
