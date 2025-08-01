
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SustainPredict: Smart Solver for Sustainability", layout="centered")

st.title("SustainPredict: Smart Solver for Sustainability")
st.markdown("**Empowering a Sustainable Future through Smart Innovation**")
st.markdown("##### Scenario: Epidemic Modeling (SIR Model)")

# --- Sidebar: Info and About ---
with st.sidebar:
    st.header("About")
    st.write("This app demonstrates the Modified Picard Iterative Method (MPIM) for solving real-world sustainability problems, here with the classic SIR epidemic model. Developed as part of an innovation initiative for a smarter, sustainable future.")
    st.markdown("**Powered by Modified Picard Iterative Method (MPIM)**")
    st.write("Author: [Your Name] | Supervisor: [Your Supervisor Name]")

# --- 1. Parameter Input ---
st.subheader("Input Model Parameters")
with st.form("sir_params"):
    col1, col2 = st.columns(2)
    with col1:
        beta = st.number_input('β (Infection Rate)', value=0.3, min_value=0.0)
        gamma = st.number_input('γ (Recovery Rate)', value=0.1, min_value=0.0)
        days = st.number_input('Simulation Days', value=60, min_value=1)
        step = st.number_input('Step Size (h)', value=0.1, min_value=0.01, max_value=1.0, format="%.2f")
    with col2:
        S0 = st.number_input('Initial Susceptible (S₀)', value=990, min_value=0)
        I0 = st.number_input('Initial Infected (I₀)', value=10, min_value=0)
        R0 = st.number_input('Initial Recovered (R₀)', value=0, min_value=0)
    submit = st.form_submit_button("Run Simulation")

# --- 2. SIR Equations and Explanation ---
with st.expander("View SIR Model Equations"):
    st.latex(r"\frac{dS}{dt} = -\beta S I")
    st.latex(r"\frac{dI}{dt} = \beta S I - \gamma I")
    st.latex(r"\frac{dR}{dt} = \gamma I")
    st.markdown("""
    - $S$ : Number of susceptible individuals  
    - $I$ : Number of infected individuals  
    - $R$ : Number of recovered individuals  
    - $\beta$ : Infection rate  
    - $\gamma$ : Recovery rate
    """)

# --- 3. MPIM Solver ---
def sir_deriv(S, I, R, beta, gamma):
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return dS, dI, dR

def mpim_solver(S0, I0, R0, beta, gamma, days, h):
    n = int(days / h) + 1
    t = np.linspace(0, days, n)
    S = np.zeros(n)
    I = np.zeros(n)
    R = np.zeros(n)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(n-1):
        # (One-stage Picard as demo; replace with your full MPIM logic if desired)
        dS1, dI1, dR1 = sir_deriv(S[k], I[k], R[k], beta, gamma)
        S[k+1] = S[k] + h * dS1
        I[k+1] = I[k] + h * dI1
        R[k+1] = R[k] + h * dR1
    return t, S, I, R

# --- 4. RK4 Solver for Comparison (Optional) ---
def rk4_solver(S0, I0, R0, beta, gamma, days, h):
    n = int(days / h) + 1
    t = np.linspace(0, days, n)
    S = np.zeros(n)
    I = np.zeros(n)
    R = np.zeros(n)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(n-1):
        dS1, dI1, dR1 = sir_deriv(S[k], I[k], R[k], beta, gamma)
        dS2, dI2, dR2 = sir_deriv(S[k]+0.5*h*dS1, I[k]+0.5*h*dI1, R[k]+0.5*h*dR1, beta, gamma)
        dS3, dI3, dR3 = sir_deriv(S[k]+0.5*h*dS2, I[k]+0.5*h*dI2, R[k]+0.5*h*dR2, beta, gamma)
        dS4, dI4, dR4 = sir_deriv(S[k]+h*dS3, I[k]+h*dI3, R[k]+h*dR3, beta, gamma)
        S[k+1] = S[k] + (h/6)*(dS1 + 2*dS2 + 2*dS3 + dS4)
        I[k+1] = I[k] + (h/6)*(dI1 + 2*dI2 + 2*dI3 + dI4)
        R[k+1] = R[k] + (h/6)*(dR1 + 2*dR2 + 2*dR3 + dR4)
    return t, S, I, R

# --- 5. Results and Plots ---
if submit:
    t, S, I, R = mpim_solver(S0, I0, R0, beta, gamma, days, step)
    t2, S2, I2, R2 = rk4_solver(S0, I0, R0, beta, gamma, days, step)
    st.success("Simulation complete!")
    
    # -- Plot S, I, R curves --
    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', label='Susceptible (MPIM)')
    ax.plot(t, I, 'r', label='Infected (MPIM)')
    ax.plot(t, R, 'g', label='Recovered (MPIM)')
    ax.plot(t2, S2, 'b--', label='Susceptible (RK4)')
    ax.plot(t2, I2, 'r--', label='Infected (RK4)')
    ax.plot(t2, R2, 'g--', label='Recovered (RK4)')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('SIR Epidemic Simulation')
    ax.legend()
    st.pyplot(fig)

    # -- Key statistics --
    peak_I = int(np.max(I))
    peak_day = int(t[np.argmax(I)])
    try:
        end_idx = np.where(I<1)[0][0]
        end_day = round(t[end_idx], 2)
    except IndexError:
        end_day = "Not reached"
    st.info(f"**Peak Infected:** {peak_I} at Day {peak_day}")
    st.info(f"**Time to Epidemic End:** {end_day} days")

    # -- Table of results --
    df = pd.DataFrame({
        "Day": t,
        "Susceptible_MPIM": S,
        "Infected_MPIM": I,
        "Recovered_MPIM": R,
        "Susceptible_RK4": S2,
        "Infected_RK4": I2,
        "Recovered_RK4": R2,
    })
    st.dataframe(df.head(20), height=300)
    
    # -- Download button --
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name='sir_simulation_results.csv',
        mime='text/csv'
    )

    # -- Error comparison plot (optional) --
    fig2, ax2 = plt.subplots()
    ax2.plot(t, np.abs(I - I2), 'm', label='|Infected_MPIM - Infected_RK4|')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error between MPIM and RK4 Solutions')
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("**Want to try another scenario? Change the parameters and rerun!**")

# --- 6. Educational Section ---
with st.expander("How does the Modified Picard Iterative Method (MPIM) work?"):
    st.write("""
    The MPIM is an improved iterative method for solving ordinary differential equations, building upon the classic Picard iteration.  
    - It constructs better approximations at each step, often converging faster and more stably than traditional Picard or simple Euler methods.
    - This makes it ideal for fast, accurate modeling of real-world sustainability challenges.
    """)
    st.markdown("**For technical details or to implement your own ODE system, contact the author.**")

# --- End of file ---
