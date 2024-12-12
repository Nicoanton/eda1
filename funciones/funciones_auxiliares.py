import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

def tipos_datos(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función tipos_datos:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Esta función recibe un conjunto de datos (DataFrame) y proporciona un resumen de los 
        tipos de datos y los valores únicos de cada columna del DataFrame. Si una columna tiene más de 30 valores 
        únicos, se muestra un mensaje indicando que hay más de 30 valores.
        - Inputs :
            -- dataset : Pandas DataFrame que contiene los datos a procesar.

        - Return : 
            None
            La función no retorna ningún valor. Su propósito es imprimir el tipo de datos y los valores únicos 
            de cada columna.
    '''

    # Número de caracteres posibles dentro del nombre de la variable
    ancho_variable = 30 

    # Establece un espacio de 12 caracteres para los tipos de datos
    ancho_tipo = 12

    for variable in dataset.columns:
    # Obtenemos los valores únicos de cada variable
        contenido_variable = dataset[variable].unique()
    # Obtenemos el tipo de variable 
        tipo_variable = dataset[variable].dtype
    # Generamos un límite, en este caso fue de 30 debido a que visualizar tantos valores en pantalla no sería lo más óptimo.
        if len(contenido_variable) > 30:
            contenido = "Más de 30 valores"
        
    # Cada elemento de la variable pasa a str, y los une en una soloa línea. Los cuales los une el join y los separa la ','
        else:
            contenido = ", ".join(map(str, contenido_variable))
    
    # ljust(), alinea las respectivas variables
        print(f"{variable.ljust(ancho_variable)} {str(tipo_variable).ljust(ancho_tipo)} Contenido: {contenido}")

def nulos_columna(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función nulos_columna:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Funcion que recibe un dataset y devuelve la cantidad de nulos por columna de la base de
        datos
        - Inputs:
            -- dataset: Pandas DataFrame que contiene los datos a procesar.
        - Return:
            -- df_nulos: DatFrame de Pandas con la cantidad de nulos por columna y el porcentaje de nulos por 
            columna
    '''
    # Cantidad de nulos en la columna
    nulos = dataset.isnull().sum().sort_values(ascending=False)
    # Porcentaje de nulos en la columna
    porcentaje_nulos = (nulos / len(dataset)) * 100
    # Creación del DataFrame, conteniendo los nulos por columna
    df_nulos = pd.DataFrame(nulos, columns=['nulos_columnas'])
    # Se añade el porcentaje de nulos al DataFrame
    df_nulos['porcentaje_columnas'] = porcentaje_nulos
    return df_nulos

def nulos_filas(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función nulos_filas:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Funcion que recibe un dataset y devuelve la cantidad de nulos por fila de la base de
        datos
        - Inputs:
            -- dataset: Pandas DataFrame que contiene los datos a procesar.
        - Return:
            -- df_nulos: DatFrame de Pandas con la cantidad de nulos por fila y el porcentaje de nulos por fila
    '''
    # Cantidad de nulos en la columna
    nulos = dataset.isnull().sum(axis = 1).sort_values(ascending=False)
    # Creación del DataFrame, conteniendo los nulos por columna
    df_nulos = pd.DataFrame(nulos, columns=['nulos_filas'])

    df_nulos['porcentaje_filas'] = df_nulos['nulos_filas']/dataset.shape[1]

    return df_nulos

def valores_booleanos(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función valores_booleanos:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Funciín que recibe un dataset y transforma las con el contenido deseado
        - Inputs : 
            -- dataset: Pandas DataFrame que contiene los datos a procesar.
        - Return:
            --dataset:  Pandas DataFrame modificado con los nuevos valores
    '''
    for column in dataset.columns:
        dataset[column] = (
            dataset[column]
            .fillna(0)  # Maneja valores nulos
            .astype(str)  # Asegura que los valores sean cadenas para el reemplazo
            .replace({
                'yes': 1, 'Y': 1, 'YES': 1, 'Yes': 1,
                'no': 0, 'No': 0, 'N': 0, 'NO': 0,
                'si': 1, 'Si': 1, 'SI': 1,  # Casos adicionales
                'nope': 0, 'null': 0, 'n/a': 0  # Otros valores según sea necesario
            })
            .astype(int)  # Asegura la conversión final a entero
        )
    return dataset

def clasificar_variables(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función clasificar_variables:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Funcion que recibe un dataset y devuelve una lista respectiva para cada tipo de variable
        (Categórica, Continua, Booleana y No clasificada)
        - Inputs:
            -- dataset : Pandas dataframe que contiene los datos
        - Return : 
            -- 1: la ejecución es incorrecta
            -- lista_var_bool: lista con los nombres de las variables booleanas del dataset de entrada, con valores
            unicos con una longitud de dos, que sean del tipo booleano y que presenten valores 'yes','no','n' & 'y' .
            -- lista_var_cat: lista con los nombres de las variables categóricas del dataset de entrada, con valores
            de tipo object o tipo categorical.
            -- lista_var_con: lista con los nombres de las variables continuas del dataset de entrada, con valores 
            de tipo float o con una longitud de valores unicos mayor a dos. 
            -- lista_var_no_clasificadas: lista con los nombres de las variables no clasificadas del dataset de 
            entrada, que no cumplen con los aspectos anteriormente mencionadas de las demás listas. 
    '''
    
    if dataset is None:
        # Resultante al no brindar ningun DataFrame
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    # Listas para cada tipo de variable
    lista_var_bool = []
    lista_var_cat = []
    lista_var_con = []
    lista_var_no_clasificadas = []
    
    for columna in dataset.columns:
        # Valores unicos por columna sin los NAs
        valores_unicos = dataset[columna].dropna().unique()
        # Trato de mayusculas
        valores_lower = set(val.lower() for val in valores_unicos if isinstance(val, str))
        
        # Variables booleanas
        if (len(valores_unicos) == 2 and
            (valores_lower <= {"yes", "no", "n", "y"} or
             set(valores_unicos) <= {0, 1} or 
             pd.api.types.is_bool_dtype(dataset[columna]))):
            lista_var_bool.append(columna)
        
        # Variables continuas
        elif pd.api.types.is_float_dtype(dataset[columna]) and len(valores_unicos) > 2:
            lista_var_con.append(columna)
        
        # Variables categóricas
        elif pd.api.types.is_object_dtype(dataset[columna]) or pd.api.types.is_categorical_dtype(dataset[columna]):
            lista_var_cat.append(columna)
        
        elif set(valores_unicos).issubset({1, 2, 3}):
            lista_var_cat.append(columna)
        
        # Variables no clasificadas
        else:
            lista_var_no_clasificadas.append(columna) 

    # Calcula la cantidad de cada tipo de variable
    c_v_b = len(lista_var_bool)
    c_v_ca = len(lista_var_cat)
    c_v_co = len(lista_var_con)
    c_v_f = len(lista_var_no_clasificadas)

    print("Variables Booleanas:", c_v_b, lista_var_bool)
    print('============================================================================================================================================================================')
    print("Variables Categóricas:", c_v_ca, lista_var_cat)
    print('============================================================================================================================================================================')
    print("Variables Continuas:", c_v_co, lista_var_con)
    print('============================================================================================================================================================================')
    print("Variables no clasificadas:", c_v_f, lista_var_no_clasificadas)

    return lista_var_bool, lista_var_cat, lista_var_con, lista_var_no_clasificadas

def nueva_clasificar_variables(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función nueva_clasificar_variables:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Funcion que recibe un dataset y devuelve una lista respectiva actualizada para cada 
        tipo de variable (Categórica, Continua, Booleana y No clasificada)
        - Inputs:
            -- dataset : Pandas dataframe que contiene los datos
        - Return : 
            -- 1: la ejecución es incorrecta
            -- lista_var_bool: lista con los nombres de las variables booleanas actualizadas del dataset de 
            entrada, con valores unicos con una longitud de dos, que sean del tipo booleano y que presenten 
            valores 'yes','no','n' & 'y' .
            -- lista_var_cat: lista con los nombres de las variables categóricas actualizadas del dataset de 
            entrada, con valores de tipo object o tipo categorical.
            -- lista_var_con: lista con los nombres de las variables continuas actualizadas del dataset de 
            entrada, con valores de tipo float o con una longitud de valores unicos mayor a dos. 
            -- lista_var_no_clasificadas: lista con los nombres de las variables no clasificadas actualizadas del 
            dataset de entrada, que no cumplen con los aspectos anteriormente mencionadas de las demás listas. 
    '''
    if dataset is None:
        # Resultante al no brindar ningun DataFrame
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    # Listas para cada tipo de variable
    lista_var_bool = []
    lista_var_cat = []
    lista_var_con = []
    lista_var_no_clasificadas = []
    
    for columna in dataset.columns:
        # Valores unicos por columna sin los NAs
        valores_unicos = dataset[columna].dropna().unique()
        # Trato de mayusculas
        valores_lower = set(val.lower() for val in valores_unicos if isinstance(val, str))
        
        # Variables booleanas
        if (len(valores_unicos) == 2 and
            (valores_lower <= {"yes", "no", "n", "y"} or
             set(valores_unicos) <= {0, 1} or 
             pd.api.types.is_bool_dtype(dataset[columna]))):
            lista_var_bool.append(columna)
        
        # Variables continuas
        elif pd.api.types.is_float_dtype(dataset[columna]) and len(valores_unicos) > 2:
            lista_var_con.append(columna)
        
        # Variables categóricas
        elif pd.api.types.is_object_dtype(dataset[columna]) or pd.api.types.is_categorical_dtype(dataset[columna]):
            lista_var_cat.append(columna)
        
        elif set(valores_unicos).issubset({1, 2, 3}):
            lista_var_cat.append(columna)
        
        # Variables no clasificadas
        else:
            lista_var_no_clasificadas.append(columna) 
    # Agregar las variables de la categoría de variables no clasificadas, a la lista deseada
    for variable in lista_var_no_clasificadas[:]:
        if variable in ['CNT_CHILDREN', 'NWEEKDAY_PROCESS_START']:
            lista_var_cat.append(variable)
        else:
            lista_var_con.append(variable)
    # Eliminar la variable no clasificada, por asignacion a otra lista
    lista_var_no_clasificadas = []

    # Cantidad de cada tipo de variable
    c_v_b = len(lista_var_bool)
    c_v_ca = len(lista_var_cat)
    c_v_co = len(lista_var_con)
    c_v_f = len(lista_var_no_clasificadas)

    print("Variables Booleanas:", c_v_b, lista_var_bool)
    print('============================================================================================================================================================================')
    print("Variables Categóricas:", c_v_ca, lista_var_cat)
    print('============================================================================================================================================================================')
    print("Variables Continuas:", c_v_co, lista_var_con)
    print('=============================================================================================================================================================================')
    print("Variables no clasificadas:", c_v_f, lista_var_no_clasificadas)

    return lista_var_bool, lista_var_cat, lista_var_con, lista_var_no_clasificadas

def plot_feature(df, col_name, isContinuous, target):
    """
    ----------------------------------------------------------------------------------------------------------
    Funcion plot_feature:
    ----------------------------------------------------------------------------------------------------------
        - Descripción :Visualizar una variable con y sin facetas respecto a la variable objetivo,ajustando para 
        casos con valores negativos.
        - Input: 
            -- df: DataFrame.
            -- col_name: Nombre de la columna.
            -- isContinuous: True si la variable es continua, False si no.
            -- target: Variable objetivo.
        - Output : 
            -- Si isContinuous:
                - Histograma
                - Boxplot
            -- No isContinuous:
                - Gráfico de barras
                - Gráfico de barras
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Crear figura para los gráficos
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=100)
    count_null = df[col_name].isnull().sum()

    # Manejar variables continuas
    if isContinuous:
        # Analizar valores negativos
        has_negatives = (df[col_name] < 0).any()
        
        if has_negatives:
            # Crear una columna temporal con valores absolutos
            df[f'{col_name}_abs'] = df[col_name].abs()

            # Graficar valores absolutos en histograma
            valid_data = df.loc[df[col_name].notnull(), f'{col_name}_abs']
            sns.histplot(valid_data, kde=False, ax=ax1)
            ax1.set_xlabel(f'{col_name} (Absolute Values)')
        else:
            # Si no hay negativos, graficar los datos originales
            valid_data = df.loc[df[col_name].notnull(), col_name]
            sns.histplot(valid_data, kde=False, ax=ax1)
            ax1.set_xlabel(col_name)

        # Boxplot con valores absolutos (si había negativos) o valores originales
        if has_negatives:
            sns.boxplot(x=f'{col_name}_abs', y=target, data=df, ax=ax2, orient='h')
            ax2.set_xlabel(f'{col_name} (Absolute Values)')
        else:
            sns.boxplot(x=col_name, y=target, data=df, ax=ax2, orient='h')
            ax2.set_xlabel(col_name)

        # Eliminar columna temporal si fue creada
        if has_negatives:
            df.drop(columns=[f'{col_name}_abs'], inplace=True)

    # Manejar variables categóricas
    else:
        # Convertir booleanos a categorías si es necesario
        if df[col_name].dtype == 'bool':
            df[col_name] = df[col_name].astype('category')

        # Contar los valores de la columna y calcular porcentajes
        counts = df[col_name].value_counts(dropna=False)
        percentages = (counts / len(df) * 100).round(1)

        # Crear gráfico de barras
        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            color='#5975A4',
            saturation=1,
            ax=ax1
        )

        # Añadir porcentajes encima de las barras
        for i, (count, perc) in enumerate(zip(counts.values, percentages.values)):
            ax1.text(i, count, f'{perc}%', ha='center', va='bottom', fontsize=10)

        # Ajustar etiquetas
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)

        # Calcular proporciones para la variable objetivo
        data = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
        data.columns = [col_name, target, 'proportion']

        # Crear gráfico de barras de proporciones
        sns.barplot(x=col_name, y='proportion', hue=target, data=data, saturation=1, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel(f'{target} Fraction')
        ax2.set_title(f'Relationship with {target}')

    # Títulos y ajustes finales
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of {col_name}\n(Missing: {count_null} values, {round(count_null / len(df) * 100, 1)}%)')
    ax2.set_ylabel(target)
    ax2.set_title(f'{col_name} vs {target}')
    plt.tight_layout()
    plt.show()

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    """
    ----------------------------------------------------------------------------------------------------------
    Funcion get_deviation_of_mean_perc:
    ----------------------------------------------------------------------------------------------------------
        - Descripción: Devuelve el porcentaje de valores que exceden del intervalo de confianza.
        - Input:
            -- pd_loan: DataFrame
            -- ist_var_continuous: lista de variables continuas
            -- target: variable objetivo
            -- multiplier: factor multiplicador para el intervalo de confianza
        - Return:
            -- pd_final con proporciones para TARGET 0 y 1 en una sola fila por variable
    """
    pd_final = pd.DataFrame()

    # Reseteamos el índice si lo habías cambiado previamente con set_index
    pd_loan = pd_loan.reset_index()

    for i in list_var_continuous:
        # Calcular la media y desviación estándar
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        # Porcentaje de valores dentro y fuera del intervalo de confianza
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size / size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size / size_s
        
        # Si existen valores fuera del intervalo de confianza
        if perc_excess > 0:
            # Crear el DataFrame con las proporciones de los valores fuera del intervalo
            pd_concat_percent = pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)] \
                .value_counts(normalize=True).reset_index()
            
            pd_concat_percent.columns = [target, 'proportion']  # Asegurarse de tener nombres de columnas correctos
            pd_concat_percent['variable'] = i
            # Recalcular sum_outlier_values correctamente
            outlier_values = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['sum_outlier_values'] = outlier_values
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess

            # Reorganizar los datos para que en una fila tengamos las proporciones de TARGET 0 y TARGET 1
            pd_concat_percent_pivot = pd_concat_percent.pivot(index='variable', columns=target, values='proportion')

            # Asegurarse de que haya una columna para cada valor de TARGET (0 y 1)
            pd_concat_percent_pivot = pd_concat_percent_pivot.fillna(0)  # Si falta algún valor, rellenamos con 0

            # Añadir las columnas adicionales necesarias
            pd_concat_percent_pivot['sum_outlier_values'] = outlier_values  # Ahora calculamos correctamente los outliers
            pd_concat_percent_pivot['porcentaje_sum_null_values'] = perc_excess

            # Aplanar el DataFrame para que sea una sola fila por variable
            pd_concat_percent_pivot = pd_concat_percent_pivot.reset_index()

            # Concatenar con el DataFrame final
            pd_final = pd.concat([pd_final, pd_concat_percent_pivot], axis=0).reset_index(drop=True)

    # Si no existen valores fuera del intervalo
    if pd_final.empty:
        print('No existen variables con valores fuera del intervalo de confianza')

    # Eliminar la columna 'TARGET' si sigue apareciendo de forma no deseada
    pd_final.columns.name = None  # Esto eliminará el nombre 'TARGET' en las columnas.
        
    return pd_final

def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):

    """
    ----------------------------------------------------------------------------------------------------------
    Función get_corr_matrix:
    ----------------------------------------------------------------------------------------------------------
        - Descripción: Genera una matriz de correlación visualizada como un mapa de calor (heatmap) utilizando 
        eaborn, y permite elegir el método de cálculo de correlación.
        - Inputs:
            -- dataset: DataFrame que contiene los datos para calcular las correlaciones.
            -- metodo: Método para calcular la correlación. Puede ser 'pearson' (por defecto), 'spearman', o 'kendall'.
            -- size_figure: Lista con las dimensiones del gráfico, por defecto [10, 8].
        - Return:
            -- 0: Indica que la función ha terminado correctamente.
            -- Mensaje de error si el `dataset` no es proporcionado.
    """

    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

def get_percent_null_values_target(pd_loan, list_var_continuous, target):
    """
    ----------------------------------------------------------------------------------------------------------
    Función get_percent_null_values_target:
    ----------------------------------------------------------------------------------------------------------
        - Descripción: Analiza variables continuas para calcular la proporción de valores nulos y su relación 
        con las categorías de la variable objetivo (`target`). Genera un DataFrame con información detallada 
        para cada variable con valores nulos.
        - Inputs:
            -- pd_loan: DataFrame que contiene los datos a procesar.
            -- list_var_continuous: Lista de nombres de las variables continuas a analizar.
            -- target: Nombre de la variable objetivo en el DataFrame.
        - Return:
            -- pd_final: DataFrame con las siguientes columnas:
            - `Category_0`, `Category_1`, ...: Proporciones de cada categoría del `target` cuando la variable
              continua es nula.
            - `variable`: Nombre de la variable continua.
            - `sum_null_values`: Cantidad de valores nulos en la variable continua.
            - `porcentaje_sum_null_values`: Porcentaje de valores nulos en la variable continua respecto 
              al total de filas.
    ----------------------------------------------------------------------------------------------------------
    """

    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum() > 0:
            # Obtener las proporciones de las categorías del target donde la variable es nula
            value_counts = pd_loan[target][pd_loan[i].isnull()].value_counts(normalize=True)
            
            # Verificar si hay alguna categoría en el target cuando la variable está nula
            if not value_counts.empty:
                # Convertir el value_counts en un DataFrame con categorías y sus proporciones
                pd_concat_percent = value_counts.reset_index()
                pd_concat_percent.columns = ['Category', 'Proportion']
                
                # Crear un DataFrame con las columnas Category_0, Category_1, ...
                category_columns = [f"Category_{k}" for k in range(len(pd_concat_percent))]
                pd_concat_percent = pd_concat_percent.set_index('Category').T
                pd_concat_percent.columns = category_columns
                
                # Agregar columnas adicionales con información
                pd_concat_percent['variable'] = i
                pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
                pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum() / pd_loan.shape[0]
                
                # Concatenar los resultados al DataFrame final
                pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            else:
                # Si no hay categorías, solo agregar la información básica
                pd_concat_percent = pd.DataFrame({
                    'variable': [i],
                    'sum_null_values': [pd_loan[i].isnull().sum()],
                    'porcentaje_sum_null_values': [pd_loan[i].isnull().sum() / pd_loan.shape[0]],
                    'Category_0': [None]
                })
                pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
    
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

def cramers_v(confusion_matrix):
    """
    ----------------------------------------------------------------------------------------------------------
    Función cramers_v:
    ----------------------------------------------------------------------------------------------------------
        - Descripción: Calcula la estadística de asociación de **Cramér's V** para medir la relación entre dos 
        variables categóricas. Utiliza una corrección propuesta por Bergsma y Wicher (2013) para manejar 
      tablas de contingencia de cualquier tamaño.
        - Inputs:
            -- confusion_matrix: DataFrame que representa una tabla de contingencia, creada típicamente con 
            `pd.crosstab()`.
        - Return:
            -- Cramér's V: Valor entre 0 y 1 que indica la fuerza de la asociación entre las dos variables:
            - 0 indica independencia (sin relación).
            - 1 indica una asociación perfecta.
    ----------------------------------------------------------------------------------------------------------
    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def c_y_t_variables_t(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función c_y_t_variables_t:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Función que recibe un dataset y clasifica sus variables en categóricas, continuas, 
        booleanas y no clasificadas, realizando las transformaciones necesarias para actualizar los tipos de 
        datos en el DataFrame.
        - Inputs:
            -- dataset : Pandas DataFrame que contiene los datos.
        - Return : 
            -- 1: la ejecución es incorrecta.
            -- dataset : Pandas DataFrame actualizado con los tipos de datos transformados.
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    # Listas para cada tipo de variable
    lista_var_bool = []
    lista_var_cat = []
    lista_var_con = []
    lista_var_no_clasificadas = []
    
    for columna in dataset.columns:
        # Valores únicos por columna sin los NAs
        valores_unicos = dataset[columna].dropna().unique()
        # Trato de mayúsculas
        valores_lower = set(val.lower() for val in valores_unicos if isinstance(val, str))
        
        # Variables booleanas
        if (len(valores_unicos) == 2 and
            (valores_lower <= {"yes", "no", "n", "y"} or
             set(valores_unicos) <= {0, 1} or 
             pd.api.types.is_bool_dtype(dataset[columna]))):
            lista_var_bool.append(columna)
        
        # Variables continuas
        elif pd.api.types.is_float_dtype(dataset[columna]) and len(valores_unicos) > 2:
            lista_var_con.append(columna)
        
        # Variables categóricas
        elif pd.api.types.is_object_dtype(dataset[columna]) or pd.api.types.is_categorical_dtype(dataset[columna]):
            lista_var_cat.append(columna)
        
        elif set(valores_unicos).issubset({1, 2, 3}):
            lista_var_cat.append(columna)
        
        # Variables no clasificadas
        else:
            lista_var_no_clasificadas.append(columna)
    
    # Reasignar variables no clasificadas si es necesario
    for variable in lista_var_no_clasificadas[:]:
        if variable in ['CNT_CHILDREN', 'NWEEKDAY_PROCESS_START']:
            lista_var_cat.append(variable)
        else:
            lista_var_con.append(variable)
    lista_var_no_clasificadas = []  # Vaciar la lista tras reasignar

    # Transformaciones de tipos
    try:
        dataset[lista_var_cat] = dataset[lista_var_cat].astype("category")
        dataset[lista_var_con] = dataset[lista_var_con].astype(float)
        dataset[lista_var_con] = dataset[lista_var_con].apply(pd.to_numeric, errors='coerce')
        if 'TARGET' in dataset.columns:
            dataset['TARGET'] = dataset['TARGET'].astype(int)
    except Exception as e:
        print("Error durante la transformación de tipos:", e)
        return 1

    # Imprimir resumen
    print("Variables Booleanas:", len(lista_var_bool), lista_var_bool)
    print('============================================================================================================================================================================')
    print("Variables Categóricas:", len(lista_var_cat), lista_var_cat)
    print('============================================================================================================================================================================')
    print("Variables Continuas:", len(lista_var_con), lista_var_con)
    print('============================================================================================================================================================================')
    print("Variables no clasificadas:", len(lista_var_no_clasificadas), lista_var_no_clasificadas)
    
    return dataset

def c_y_t_variables_tt(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función c_y_t_variables_tt:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Función que recibe un dataset y clasifica sus variables en categóricas, continuas, 
        booleanas y no clasificadas, realizando las transformaciones necesarias para actualizar los tipos de 
        datos en el DataFrame.
        - Inputs:
            -- dataset : Pandas DataFrame que contiene los datos.
        - Return : 
            -- 1: la ejecución es incorrecta.
            -- dataset : Pandas DataFrame actualizado con los tipos de datos transformados.
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    # Listas para cada tipo de variable
    lista_var_bool = []
    lista_var_cat = []
    lista_var_con = []
    lista_var_no_clasificadas = []
    
    for columna in dataset.columns:
        # Valores únicos por columna sin los NAs
        valores_unicos = dataset[columna].dropna().unique()
        # Trato de mayúsculas
        valores_lower = set(val.lower() for val in valores_unicos if isinstance(val, str))
        
        # Variables booleanas
        if (len(valores_unicos) == 2 and
            (valores_lower <= {"yes", "no", "n", "y"} or
             set(valores_unicos) <= {0, 1} or 
             pd.api.types.is_bool_dtype(dataset[columna]))):
            lista_var_bool.append(columna)
        
        # Variables continuas
        elif pd.api.types.is_float_dtype(dataset[columna]) and len(valores_unicos) > 2:
            lista_var_con.append(columna)
        
        # Variables categóricas
        elif pd.api.types.is_object_dtype(dataset[columna]) or pd.api.types.is_categorical_dtype(dataset[columna]):
            lista_var_cat.append(columna)
        
        elif set(valores_unicos).issubset({1, 2, 3}):
            lista_var_cat.append(columna)
        
        # Variables no clasificadas
        else:
            lista_var_no_clasificadas.append(columna)
    
    # Reasignar variables no clasificadas si es necesario
    for variable in lista_var_no_clasificadas[:]:
        if variable in ['CNT_CHILDREN', 'NWEEKDAY_PROCESS_START']:
            lista_var_cat.append(variable)
        else:
            lista_var_con.append(variable)
    lista_var_no_clasificadas = []  # Vaciar la lista tras reasignar

    for variable in lista_var_cat[:]:
        if variable == 'FLAG_MOBIL':
            lista_var_bool.append(variable)
            lista_var_cat.remove(variable)
    # Transformaciones de tipos
    try:
        dataset[lista_var_cat] = dataset[lista_var_cat].astype("category")
        dataset[lista_var_con] = dataset[lista_var_con].astype(float)
        dataset[lista_var_con] = dataset[lista_var_con].apply(pd.to_numeric, errors='coerce')
        if 'TARGET' in dataset.columns:
            dataset['TARGET'] = dataset['TARGET'].astype(int)
    except Exception as e:
        print("Error durante la transformación de tipos:", e)
        return 1

    # Imprimir resumen
    print("Variables Booleanas:", len(lista_var_bool), lista_var_bool)
    print('============================================================================================================================================================================')
    print("Variables Categóricas:", len(lista_var_cat), lista_var_cat)
    print('============================================================================================================================================================================')
    print("Variables Continuas:", len(lista_var_con), lista_var_con)
    print('============================================================================================================================================================================')
    print("Variables no clasificadas:", len(lista_var_no_clasificadas), lista_var_no_clasificadas)
    
    return dataset

def tipo_encoding(df, lista_col_cat, ordinal_orders):
    """
    ----------------------------------------------------------------------------------------------------------
    Función tipo_encoding:
    ----------------------------------------------------------------------------------------------------------
        - Descripción:
            Clasifica las variables categóricas en listas para One-Hot Encoding, Ordinal Encoding y 
            Target Encoding basándose en el número de categorías únicas y un diccionario de órdenes.
        
        - Inputs:
            -- df: (DataFrame) DataFrame con las columnas a clasificar.
            -- lista_col_cat: (list) Lista de nombres de columnas categóricas.
            -- ordinal_orders: (dict) Diccionario donde:
                  - Clave: Nombre de la columna.
                  - Valor: Lista que define el orden preestablecido de las categorías para Ordinal Encoding.
        
        - Output:
            -- (list, list, list) Tres listas: 
               - `ohe`: Variables para One-Hot Encoding.
               - `ordinal`: Variables para Ordinal Encoding.
               - `targ`: Variables para Target Encoding.
    ----------------------------------------------------------------------------------------------------------
    """
    # Inicializar las listas
    ohe = []       # Lista para variables de One-Hot Encoding
    ordinal = []   # Lista para variables de Ordinal Encoding
    targ = []      # Lista para variables de Target Encoding

    # Iterar sobre las variables categóricas
    for variable in lista_col_cat:
        try:
            # Obtener los valores únicos y su cantidad desde el DataFrame
            valores_unicos = df[variable].unique()
            cantidad = len(valores_unicos)

            # Clasificar según el tipo de codificación
            if variable in ordinal_orders:  # Si la variable está en el diccionario de órdenes
                ordinal.append(variable)
            elif cantidad > 10:  # Más de 10 categorías únicas
                targ.append(variable)
            else:  # Menos o igual a 10 categorías únicas
                ohe.append(variable)
        except Exception as e:
            print(f"Error procesando {variable}: {e}")

    def print_in_blocks(variable_list, title):
        print(f"\n{title}:")
        for i in range(0, len(variable_list), 3):
            print(", ".join(variable_list[i:i+3]))
    
    # Imprimir las listas en bloques de 3 variables
    print_in_blocks(ohe, "Variables (Nominales) - One-Hot Encoding")
    print_in_blocks(ordinal, "Variables (Ordinales) - Ordinal Encoding")
    print_in_blocks(targ, "Variables (Alto volumen) - Target Encoding")
    
    return ohe, ordinal, targ
    # Retornar las tres listas

def m_confusion(y_true, y_res, modelo='', class_labels=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función m_confusion:
    ----------------------------------------------------------------------------------------------------------
        - Descripción:
            Genera dos gráficos, el primero para una matriz de confusión y el segundo para una matriz de confusión
            normalizada que muestra proporciones relativas
        
        - Inputs:
            -- y_true : Valores reales de y
            -- y_res : Valores predichos de y
            -- modelo : Nombre del modelo (Con el fin de establecerlo en las gráficas)
            -- class_labels : Lista con las etiquetas de las clases a mostrar en los ejes de las matrices.

        - Output:
            -- Gráficos de matrices de confusión:
                -- Matriz de confusión estándar
                -- Matriz de confusión normalizada
    ----------------------------------------------------------------------------------------------------------
    '''
 
    # Validar las entradas
    if len(y_true) != len(y_res):
        raise ValueError("y_true y y_res deben tener la misma longitud.")
    
    # Crear las matrices de confusión
    matriz = confusion_matrix(y_true, y_res)
    matriz_normalizada = matriz.astype('float') / matriz.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Matriz de confusión
    m1 = ConfusionMatrixDisplay(matriz, display_labels=class_labels)
    m1.plot(cmap='Blues', values_format=',.0f', ax=axes[0])
    axes[0].set_title(f'Matriz de Confusión {modelo}', fontsize=14)
    
    # Matriz de confusión normalizada
    m2 = ConfusionMatrixDisplay(matriz_normalizada, display_labels=class_labels)
    m2.plot(cmap='Blues', values_format='.2%', ax=axes[1])
    axes[1].set_title(f"Matriz de Confusión (Normalizada) {modelo}", fontsize=14)
    
    # Ajustes comunes
    for ax in axes:
        ax.set_xlabel('Clase Predicha', fontsize=12)
        ax.set_ylabel('Clase Real', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Configurar el espaciado de la figura
    plt.tight_layout(pad=2.0)
    plt.show()

def curva_roc(y_true, prob_pred, model_name="Modelo"):
    """
    Calcula y grafica la curva ROC y el AUC para un modelo de clasificación binaria.
    
    Parameters:
    - y_true: array-like, etiquetas verdaderas del conjunto de prueba.
    - prob_pred: array-like, probabilidades predichas por el modelo.
    - model_name: string, nombre del modelo para la etiqueta en la gráfica.
    
    Returns:
    - roc_auc: valor del área bajo la curva (AUC) para el modelo.
    """
    # Obtener probabilidades para la clase positiva (1)
    yhat = prob_pred[:, 1] 

    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, yhat)  # Calculamos FPR y TPR

    # Calcular el área bajo la curva (AUC)
    roc_auc = auc(fpr, tpr)

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))

    # Curva ROC del modelo
    plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Línea diagonal (representa un clasificador aleatorio)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Sin Habilidad')

    # Etiquetas de los ejes
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')

    # Título
    plt.title('Curva ROC')

    # Leyenda
    plt.legend(loc='lower right')

    # Mostrar el gráfico
    plt.grid(True)
    plt.show()

    return roc_auc

def roc_threshold(model, X_test, y_test):
    """
    Calcula la curva ROC, el área bajo la curva (AUC), el G-Mean y el mejor umbral.
    Luego, grafica la curva ROC y marca el mejor umbral en la gráfica.
    
    Parameters:
    - model: Modelo entrenado que tiene el método predict_proba().
    - X_test: Características del conjunto de prueba.
    - y_test: Etiquetas del conjunto de prueba.
    
    Returns:
    - best_threshold: El umbral óptimo basado en el G-Mean.
    - roc_auc: El área bajo la curva ROC.
    """
    # Obtener probabilidades para ambas clases
    prob_predictions = model.predict_proba(X_test)

    # Extraer las probabilidades solo para la clase positiva
    yhat = prob_predictions[:, 1]

    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, yhat)

    # Calcular el área bajo la curva (AUC)
    roc_auc = auc(fpr, tpr)

    # Calcular G-Mean
    gmeans = np.sqrt(tpr * (1 - fpr))  # G-Mean

    # Encontrar el índice del mejor threshold (el que maximiza el G-Mean)
    ix = np.argmax(gmeans)

    # Obtener el mejor threshold
    best_threshold = thresholds[ix]

    print(f'Best Threshold = {best_threshold:.6f}')

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))

    # Curva ROC del modelo 
    plt.plot(fpr, tpr, marker='.', color='orange', label=f'Modelo (AUC = {roc_auc:.2f})')

    # Línea diagonal (representa un clasificador aleatorio, en color azul)
    plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Sin habilidad')

    # Mostrar el mejor umbral en la gráfica (en color negro)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', s=100, label=f'Best Threshold = {best_threshold:.2f}')

    # Etiquetas de los ejes
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')

    # Título y leyenda
    plt.title('Curva ROC con Mejor Threshold')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

    # Retornar el mejor umbral y AUC
    return best_threshold, roc_auc

def curva_recall(model, X_test, y_test):
    """
    Calcula la curva Precision-Recall para el modelo, muestra la precisión para un clasificador aleatorio
    y grafica la curva Precision-Recall.
    
    Parameters:
    - model: Modelo entrenado que tiene el método predict_proba().
    - X_test: Características del conjunto de prueba.
    - y_test: Etiquetas del conjunto de prueba.
    
    Returns:
    - precision: Array con los valores de precisión para cada umbral.
    - recall: Array con los valores de recall para cada umbral.
    - thresholds: Array con los valores de los umbrales.
    """
    # Obtener las probabilidades para ambas clases
    prob_predictions = model.predict_proba(X_test)  # Obtener probabilidades para ambas clases

    # Extraer las probabilidades para la clase positiva (índice 1)
    yhat = prob_predictions[:, 1]

    # Calcular la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)

    # Calcular la precisión para un clasificador aleatorio (sin habilidad)
    no_skill = len(y_test[y_test == 1]) / len(y_test)

    # Graficar la curva Precision-Recall
    plt.figure(figsize=(8, 6))

    # Línea de no habilidad (precisión igual a la frecuencia de la clase positiva)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='blue', label='No Skill')

    # Curva Precision-Recall para el modelo
    plt.plot(recall, precision, marker='.', color='orange', label='Modelo')

    # Etiquetas de los ejes
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Título y leyenda
    plt.title('Curva Precision-Recall')
    plt.legend(loc='lower left')
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

    # Retornar los valores de precisión, recall y thresholds
    return precision, recall, thresholds

from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

def curva_recall_f(model, X_test, y_test):
    """
    Calcula la curva Precision-Recall, encuentra el mejor umbral basado en el F-Score
    y grafica la curva Precision-Recall con el mejor umbral.
    
    Parameters:
    - model: Modelo entrenado que tiene el método predict_proba().
    - X_test: Características del conjunto de prueba.
    - y_test: Etiquetas del conjunto de prueba.
    
    Returns:
    - precision: Array con los valores de precisión para cada umbral.
    - recall: Array con los valores de recall para cada umbral.
    - thresholds: Array con los valores de los umbrales.
    - best_threshold: El mejor umbral basado en el F-Score.
    - best_fscore: El mejor valor de F-Score.
    """
    # Obtener las probabilidades para ambas clases
    prob_predictions = model.predict_proba(X_test)  # Obtener probabilidades para ambas clases

    # Extraer las probabilidades para la clase positiva (índice 1)
    yhat = prob_predictions[:, 1]

    # Calcular la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)

    # Calcular el F-Score para cada umbral
    fscore = (2 * precision * recall) / (precision + recall)

    # Localizar el índice del F-Score más grande
    ix = np.argmax(fscore)
    best_threshold = thresholds[ix]
    best_fscore = fscore[ix]

    print(f'Best Threshold={best_threshold:.6f}, F-Score={best_fscore:.3f}')

    # Calcular la precisión para un clasificador aleatorio (sin habilidad)
    no_skill = len(y_test[y_test == 1]) / len(y_test)

    # Graficar la curva Precision-Recall
    plt.figure(figsize=(8, 6))

    # Línea de no habilidad (precisión igual a la frecuencia de la clase positiva)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='blue', label='No Skill')

    # Curva Precision-Recall para el modelo
    plt.plot(recall, precision, marker='.', color='orange', label='Modelo')

    # Marcar el punto con el mejor F-Score en la gráfica
    plt.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label=f'Best Threshold = {best_threshold:.2f}')

    # Etiquetas de los ejes
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Título y leyenda
    plt.title('Curva Precision-Recall con Mejor F-Score')
    plt.legend(loc='lower left')
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

    # Retornar los valores de precisión, recall, thresholds, el mejor umbral y el mejor F-Score
    return precision, recall, thresholds, best_threshold, best_fscore

def ganancia_acumulada(y_true, prob_pred):
    """
    Calcula la ganancia acumulada para cada clase y grafica la curva.
    """
    unique_classes = np.unique(y_true)
    gains = {}

    for class_label in unique_classes:
        # Para cada clase, obtén las probabilidades predichas
        prob_class = prob_pred[:, class_label] if prob_pred.ndim > 1 else prob_pred

        # Etiquetas binarizadas (1 para la clase actual, 0 para las demás)
        y_true_binary = (y_true == class_label).astype(int)

        # Ordenar las instancias según las probabilidades de la clase
        order = np.argsort(prob_class)[::-1]
        y_true_sorted = y_true_binary[order]

        # Calcular la ganancia acumulada
        cumulative_gain = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        cumulative_gain = np.insert(cumulative_gain, 0, 0)  # Insertar el valor 0 al inicio

        # Guardar los resultados
        gains[class_label] = (np.linspace(0, 1, len(cumulative_gain)), cumulative_gain)

    # Graficar la curva de ganancia acumulada para cada clase
    plt.figure(figsize=(8, 6))
    for class_label, (recall, gain) in gains.items():
        plt.plot(recall, gain, label=f'Clase {class_label}')

    # Línea base: ganancia acumulada aleatoria
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')

    # Configuración de la gráfica
    plt.title('Curva de Ganancia Acumulada')
    plt.xlabel('Porcentaje de la Muestra')
    plt.ylabel('Ganancia Acumulada')
    plt.legend()
    plt.grid(True)
    plt.show()

def curva_lift_dos(y_true, prob_predictions, class_1=1, class_0=0):
    """
    Genera la Lift Curve para las dos clases de un modelo dado.
    
    Parameters:
    - y_true: Etiquetas verdaderas del conjunto de prueba.
    - prob_predictions: Probabilidades predichas para ambas clases.
    - class_1: Etiqueta de la clase positiva (por defecto 1).
    - class_0: Etiqueta de la clase negativa (por defecto 0).
    """
    # Obtener las probabilidades para la clase positiva (class_1) y negativa (class_0)
    prob_class_1 = prob_predictions[:, class_1]
    prob_class_0 = prob_predictions[:, class_0]

    # Ordenar las instancias en función de la probabilidad de la clase positiva
    order = np.argsort(prob_class_1)[::-1]
    y_true_sorted = y_true[order]
    
    # Calcular la lift para cada porcentaje acumulado de instancias
    lift_class_1 = []
    lift_class_0 = []
    
    for percentile in np.arange(0.1, 1.1, 0.1):  # Desde el 10% hasta el 100%
        # Número de instancias a considerar en este percentile
        cutoff = int(len(y_true_sorted) * percentile)
        
        # Obtener el número de positivos y negativos en el conjunto seleccionado
        selected_positives = np.sum(y_true_sorted[:cutoff] == class_1)
        selected_negatives = np.sum(y_true_sorted[:cutoff] == class_0)
        
        # Calcular el lift para la clase positiva y negativa
        lift_class_1.append(selected_positives / (percentile * np.sum(y_true == class_1)))
        lift_class_0.append(selected_negatives / (percentile * np.sum(y_true == class_0)))
    
    # Graficar la Lift Curve
    plt.plot(np.arange(0.1, 1.1, 0.1) * 100, lift_class_1, label=f'Clase {class_1} (Positiva)', color='orange')
    plt.plot(np.arange(0.1, 1.1, 0.1) * 100, lift_class_0, label=f'Clase {class_0} (Negativa)', color='blue')
    plt.xlabel('Porcentaje de la Muestra (%)')
    plt.ylabel('Lift')
    plt.title('Curva lift para las Dos Clases')
    plt.legend()
    plt.grid(True)
    plt.show()