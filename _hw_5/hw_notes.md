1. Refazer as soluções das equações não lineares de exemplo dos slides da aulas

2. Resolver a equação da difusão (Runge-Kutta)

Observação: implementação manual do runge-kutta pode ajudar a ter mais flexibilidade do que usar um pacote pronto
    -> avanço de passo: cálculo de N+1, u(x=0) provavelmente não vai dar zero, então você corrige e impõe que seja zero, e vai para o próximo passo
    a imposição das condições de contorno derivada=zero em cada passo pode não ser possível usando um pacote
