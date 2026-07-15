# Reporte de evaluación
- Version: 1.0.0
- Clasificador EPII v.0.0

## Resumen general

El modelo clasificador fue evaluado sobre un conjunto de 77 imágenes de cámaras trampa, balanceado en 7 clases: Chucao, Gato huiña, Hued hued del sur, Pudú, Puma, Unknown y Zorzal patagónico. En esta evaluación, el modelo alcanzó una **accuracy de tarea completa de 0.455**, considerando detección + clasificación mediante etiqueta combinada. La **tasa de detección fue alta, 0.987**, lo que indica que el sistema casi siempre detectó presencia animal. Sin embargo, la **accuracy de clasificación pura fue 0.461**, mostrando que el principal problema del sistema no está en detectar animales, sino en clasificar correctamente la especie después del crop.

## Métricas principales

| Métrica                     | Resultado |
| --------------------------- | --------: |
| n evaluadas                 |        77 |
| Accuracy tarea completa     |     0.455 |
| Tasa de detección           |     0.987 |
| Accuracy clasificación pura |     0.461 |
| F1 macro                    |     0.382 |
| F1 micro                    |     0.455 |
| F1 weighted                 |     0.436 |

El modelo presenta una diferencia importante entre su desempeño interno de validación y su desempeño en evaluación externa: durante entrenamiento alcanzó aproximadamente **0.855 de val_acc**, pero en el conjunto de evaluación obtuvo **0.455 de accuracy**. Esto muestra que el desempeño observado en validación no se transfiere directamente al conjunto de test/evaluación.

## Desempeño por clase

| Clase             | Correctas / Total | Recall por clase |
| ----------------- | ----------------: | ---------------: |
| Chucao            |            3 / 11 |            0.273 |
| Gato huiña        |            2 / 11 |            0.182 |
| Hued hued del sur |            6 / 11 |            0.545 |
| Pudú              |            9 / 11 |            0.818 |
| Puma              |            4 / 11 |            0.364 |
| Unknown           |            2 / 11 |            0.182 |
| Zorzal patagónico |            9 / 11 |            0.818 |

Las clases con mejor desempeño fueron **Pudú** y **Zorzal patagónico**, ambas con 9 aciertos de 11. Luego aparece **Hued hued del sur**, con un desempeño intermedio. Las clases más débiles fueron **Gato huiña** y **Unknown**, ambas con solo 2 aciertos de 11, seguidas por **Chucao** y **Puma**.

## Distribución de predicciones

El modelo tendió a concentrar sus predicciones en pocas clases:

| Clase predicha    |  n |
| ----------------- | -: |
| Pudú              | 18 |
| Zorzal patagónico | 17 |
| Unknown           | 14 |
| Hued hued del sur | 11 |
| Puma              |  7 |
| Chucao            |  5 |
| Gato huiña        |  4 |
| No detectado      |  1 |

Esto muestra un sesgo del modelo hacia **Pudú**, **Zorzal patagónico** y **Unknown**, mientras que predice relativamente poco **Gato huiña**, **Chucao** y **Puma**. En particular, Gato huiña aparece subpredicho: solo fue predicho 4 veces, aunque tenía 11 ejemplos reales.

## Confusiones principales

Las confusiones más frecuentes fueron:

| Real              | Predicho          |  n |
| ----------------- | ----------------- | -: |
| Gato huiña        | Unknown           |  6 |
| Puma              | Pudú              |  6 |
| Chucao            | Zorzal patagónico |  4 |
| Unknown           | Hued hued del sur |  3 |
| Unknown           | Zorzal patagónico |  2 |
| Gato huiña        | Puma              |  2 |
| Chucao            | Hued hued del sur |  2 |
| Hued hued del sur | Unknown           |  2 |

El error más marcado para **Gato huiña** fue ser clasificado como **Unknown**, lo que indica que el modelo no logró separar bien esa especie de casos ambiguos o de clases fuera del conjunto principal. En **Puma**, el error dominante fue hacia **Pudú**, con 6 casos, lo que afecta fuertemente su recall. En **Chucao**, el modelo confundió varios casos con **Zorzal patagónico** y **Hued hued del sur**, lo que sugiere dificultad para separar aves pequeñas o visualmente similares en las imágenes evaluadas.

## Detección vs clasificación

La detección funcionó correctamente en casi todo el conjunto: solo hubo **1 caso como “No detectado”**. Por lo tanto, la caída de desempeño global no se explica por fallas del detector, sino principalmente por errores de clasificación del modelo ResNet50 sobre los crops generados. La confianza promedio del detector fue alta, cercana a **0.876**, mientras que la confianza promedio del clasificador fue más moderada, cercana a **0.510**, lo que es consistente con un clasificador menos seguro en este conjunto de evaluación.

## Comparación con baseline humano
_Clasificacion_

El modelo queda por debajo del rango humano observado. Mientras los evaluadores humanos se ubicaron aproximadamente entre **0.83 y 0.92 de accuracy**, el modelo obtuvo **0.455**. La diferencia es especialmente clara en clasificación pura: los humanos alcanzaron valores entre aproximadamente **0.83 y 0.93**, mientras que el modelo obtuvo **0.461**. 

_Deteccion_

En detección, en cambio, el modelo se acerca al desempeño humano, ya que obtuvo **0.987**, similar al rango alto de los evaluadores.

## Conclusión

El modelo actual detecta animales de forma consistente, pero su capacidad de clasificación por especie aún es limitada en el conjunto de evaluación. Su desempeño está dominado por errores de clasificación, especialmente en **Gato huiña**, **Unknown**, **Chucao** y **Puma**. Las clases mejor resueltas fueron **Pudú** y **Zorzal patagónico**. En términos generales, el modelo muestra una brecha importante entre validación interna y evaluación externa, con un rendimiento final sustancialmente inferior al baseline humano.
