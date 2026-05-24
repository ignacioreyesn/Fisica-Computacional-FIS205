# Simulador de Oscilaciones de Rabi y Dinámica de Qubits

**Universidad Técnica Federico Santa María** **Departamento de Física** **Curso:** Física Computacional FIS205   
**Profesores:** Dr. Ariel Norambuena & Dr. Nicolas Viaux \
**Ayudante:** Cristóbal Benavides \
**Estudiante:** Ignacio Reyes 


---

## Descripción del Proyecto

Este repositorio contiene el desarrollo de un simulador computacional para el estudio de sistemas cuánticos de dos niveles. El objetivo principal es modelar fenómenos de interacción con el ambiente, pérdida de información y decoherencia, resolviendo las ecuaciones ópticas de Bloch (OBE) mediante métodos numéricos.

Actualmente, el simulador permite visualizar la dinámica en sistema cerrado, proyectando la evolución temporal del estado cuántico sobre la **Esfera de Bloch**.

### Demostración Visual


---

## Estructura del Repositorio

El material está organizado en las siguientes carpetas:

**`Tarea_1/`**: Contiene los scripts correspondientes a las resoluciones de la primera tarea del curso.
  * `tarea1problema1.py`
  * `tarea1problema2.py`
  * `tarea1problema4.py`
    
**`Proyecto_Avance_1/`**: Contiene la primera entrega del proyecto final.
  * `Avance 1 FIS205.pdf`: Informe detallado con el marco teórico, abarcando la parametrización de estados, ecuación de Liouville y la derivación de la ecuación maestra de Lindblad.
  * `DinamicaQubit1.py`: Código fuente del simulador actual, utilizando `scipy.integrate.solve_ivp` para resolver la dinámica cerrada.
  * `simulacion_rabi.gif`: Archivo multimedia con la animación 3D generada.

---

## Requisitos y Dependencias

El simulador está desarrollado en **Python**. Para ejecutar el código correctamente, asegúrate de tener instaladas las siguientes librerías:

* `numpy`
* `scipy`
* `matplotlib`

Puedes instalarlas rápidamente usando `pip`:

```bash
pip install numpy scipy matplotlib
