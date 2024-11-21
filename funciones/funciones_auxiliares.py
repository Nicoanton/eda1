import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as ss

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
    for column in dataset.columns:
        # Verificar si la columna contiene valores susceptibles de conversión
        if dataset[column].dtypes == object:  # Procesar solo columnas tipo objeto (cadenas)
            dataset[column] = (
                dataset[column]
                .fillna(0)  # Maneja valores nulos
                .astype(str)  # Asegura que los valores sean cadenas para el reemplazo
                .replace({
                    'yes': 1, 'Y': 1, 'YES': 1, 'Yes': 1,
                    'no': 0, 'No': 0, 'N': 0, 'NO': 0,
                    'si': 1, 'Si': 1, 'SI': 1,
                    'nope': 0, 'null': 0, 'n/a': 0
                })
            )
            # Convierte a enteros si todos los valores son convertibles
            try:
                dataset[column] = dataset[column].astype(int)
            except ValueError:
                print(f"Columna '{column}' contiene valores no booleanos, no se convierte.")
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
    Visualizar una variable con y sin facetas respecto a la variable objetivo,
    ajustando para casos con valores negativos.

    - df: DataFrame.
    - col_name: Nombre de la columna.
    - isContinuous: True si la variable es continua, False si no.
    - target: Variable objetivo.
    """

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
    Devuelve el porcentaje de valores que exceden del intervalo de confianza.
    :param pd_loan: DataFrame
    :param list_var_continuous: lista de variables continuas
    :param target: variable objetivo
    :param multiplier: factor multiplicador para el intervalo de confianza
    :return: pd_final con proporciones para TARGET 0 y 1 en una sola fila por variable
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
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))