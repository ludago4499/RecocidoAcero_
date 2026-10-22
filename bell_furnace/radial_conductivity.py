
# radial_conductivity.py

SIGMA = 5.670374419e-8 


def gas_conductance(k_g, b):
    """
    Conductancia térmica del gas entre capas.

    k_g: conductividad del gas (W/m K)
    b: separación entre capas (m)
    """
    if k_g <= 0 or b <= 0:
        raise ValueError("k_g y b deben ser positivos")

    return k_g / b


def radiation_conductance(T, epsilon):
    """
    Conductancia radiativa linealizada entre dos superficies
    paralelas con la misma emisividad.

    T: temperatura media (K)
    epsilon: emisividad superficial (0 a 1)
    """
    if T <= 0:
        raise ValueError("La temperatura debe ser positiva")

    if not 0 <= epsilon <= 1:
        raise ValueError("La emisividad debe estar entre 0 y 1")

    return 4 * SIGMA * epsilon * T**3 / (2 - epsilon)


def interface_conductance(k_g, b, T, epsilon, h_contact):
    """
    Conductancia térmica equivalente entre capas.

    """
    if h_contact < 0:
        raise ValueError("h_contact no puede ser negativo")

    h_gas = gas_conductance(k_g, b)
    h_rad = radiation_conductance(T, epsilon)

    h_int = h_gas + h_rad + h_contact

    return h_int


def radial_conductivity(s, k_s, h_int):
  

 

  
    if s <= 0 or k_s <= 0 or h_int <= 0:
        raise ValueError("Los parámetros deben ser positivos")

    R_acero = s / k_s
    R_interfaz = 1 / h_int

    R_total = R_acero + R_interfaz

    k_radial = s / R_total

    return k_radial


def effective_radial_conductivity(
    s, k_s, k_g, b, T, epsilon, h_contact
):
    """
    Calcula la conductividad radial efectiva
    a partir de los mecanismos de la interfaz.
    """
    h_int = interface_conductance(
        k_g, b, T, epsilon, h_contact
    )

    return radial_conductivity(s, k_s, h_int)


# --------------------------------------------------
# PRUEBA DEL MODELO
# Valores provisionales
# --------------------------------------------------

if __name__ == "__main__":

    s = 0.0015         # Espesor de lámina (m)
    k_s = 45           # Conductividad del acero (W/m K)
    k_g = 0.20         # Conductividad del gas (W/m K)
    b = 0.00002        # Separación entre capas (m)
    T = 850            # Temperatura media (K)
    epsilon = 0.08554  # Emisividad ilustrativa

    # Dos condiciones de contacto
    h_contact_1 = 1000
    h_contact_2 = 10000

    # Cálculo de conductividades
    k_1 = effective_radial_conductivity(
        s, k_s, k_g, b, T, epsilon, h_contact_1
    )

    k_2 = effective_radial_conductivity(
        s, k_s, k_g, b, T, epsilon, h_contact_2
    )

    print("Conductividad del acero:", k_s, "W/(m K)")
    print("Conductividad radial con menor contacto:", k_1, "W/(m K)")
    print("Conductividad radial con mayor contacto:", k_2, "W/(m K)")


    assert 0 < k_1 < k_s
    assert 0 < k_2 < k_s
    assert k_2 > k_1

    print("Comprobaciones físicas superadas")