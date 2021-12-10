import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Load South Korea Covid-19 csv data file
C_pd = pd.read_csv('covid19_data.csv')

# Converting covid19 csv data into list datatype
Length = len(C_pd)
daily_infect = list(C_pd['dailyInfect'])
daily_recovery = list(C_pd['dailyRec'])
cum_infect = list(C_pd['cumInfect'])
cum_recovery = list(C_pd['cumRec'])
net_infect = list(C_pd['netInfect'])

# Figure settings
ind = np.arange(Length)
width = 0.35

# Figure 1. South Korea Daily cases of infection and recovery from 20th Jan 2020 to 31st May 2020
plt.bar(ind, daily_infect, width, label='Infection')
plt.bar(ind + width, daily_recovery, width, label='Recovery')
plt.xlabel('Time (Day)')
plt.ylabel('Population')
plt.title('Number of Daily Infection and Recovery Cases')
plt.legend(loc='best')
plt.show()

# Figure 2. South Korea cumulative infection and recovery from 20th Jan 2020 to 31st May 2020
fig, ax = plt.subplots()
plt.bar(ind, cum_infect, width, label='Cumulative Infection', color='black')
plt.bar(ind, cum_recovery, width, label='Cumulative Recovery', color='red')
plt.xlabel('Time (Day)')
plt.ylabel('Population')
plt.title('Cumulative Number of Infection and Recovery')
plt.legend(loc='best')
plt.show()

'''
Parameter values are based on the numerical results according to the following paper.
Jeong et.al, Estimation of Reproduction Number for COVID-19 in Korea
Journal of the Korean Society for Quality Management, Volume 48 Issue 3 / Pages.493-510 / 2020
'''

# initial number of infected, recovered and susceptible individuals for SIR model
i_init_sir = 1/10000
r_init_sir = 1/20000
s_init_sir = 1 - i_init_sir - r_init_sir

# parameter values for SIR model
R0_sir = 3.1
gamma_sir = 0.11
beta_sir = R0_sir * gamma_sir

# initial number of infected, recovered and susceptible individuals for SEIR model
e_init_seir = 1/5000
i_init_seir = 1/10000
r_init_seir = 1/20000
s_init_seir = 1 - e_init_seir - i_init_seir - r_init_seir

# parameter values for SEIR model
R0_seir = 3.1
raw_alpha_seir = 0.083
alpha_seir = raw_alpha_seir * 10
gamma_seir = 0.11
beta_seir = R0_seir * gamma_seir


# SIR model differential equations
def deriv(x, t, beta, gamma):
    s, i, r = x
    dsdt = -beta * s * i
    didt = beta * s * i - gamma * i
    drdt = gamma * i
    return [dsdt, didt, drdt]


# SEIR model differential equations
def seir_deriv(x, t, alpha, beta, gamma):
    s, e, i, r = x
    dsdt = -beta * s * i
    dedt = beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt = gamma * i
    return [dsdt, dedt, didt, drdt]


# Graphical representation for SIR model
def plotdata_sir(t, s, i, e=None):
    # plot the data
    fig = plt.figure(figsize=(12, 6))
    ax = [fig.add_subplot(221, axisbelow=True),
          fig.add_subplot(223),
          fig.add_subplot(122)]

    ax[0].plot(t, i, lw=3, label='Fraction Infective')
    ax[0].plot(t, r, lw=3, label='Recovered')
    ax[0].set_title('Susceptible and Recovered Populations')
    ax[0].set_xlabel('Time /days')
    ax[0].set_ylabel('Population (in 20,000)')

    ax[1].plot(t, i, lw=3, label='Infective')
    ax[1].set_title('Infectious Population')
    if e is not None: ax[1].plot(t, e, lw=3, label='Exposed')
    ax[1].set_ylim(0, 0.5)
    ax[1].set_xlabel('Time /days')
    ax[1].set_ylabel('Population (in 20,000)')

    ax[2].plot(s, i, lw=3, label='s, i trajectory')
    ax[2].plot([1 / R0_sir, 1 / R0_sir], [0, 1], '--', lw=3, label='di/dt = 0')
    ax[2].plot(s[0], i[0], '.', ms=20, label='Initial Condition')
    ax[2].plot(s[-1], i[-1], '.', ms=20, label='Final Condition')
    ax[2].set_title('State Trajectory')
    ax[2].set_aspect('equal')
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlim(0, 1.05)
    ax[2].set_xlabel('Susceptible (in 20,000)')
    ax[2].set_ylabel('Infectious (in 20,000)')

    for a in ax:
        a.grid(True)
        a.legend()

    plt.tight_layout()


# Graphical representation for SEIR model
def plotdata_seir(t, s, i, e=None):
    # plot the data
    fig = plt.figure(figsize=(12, 6))
    ax = [fig.add_subplot(221, axisbelow=True),
          fig.add_subplot(223),
          fig.add_subplot(122)]

    ax[0].plot(t, i, lw=3, label='Fraction Infective')
    ax[0].plot(t, r, lw=3, label='Recovered')
    ax[0].set_title('Susceptible and Recovered Populations')
    ax[0].set_xlabel('Time /days')
    ax[0].set_ylabel('Population (in 20,000)')

    ax[1].plot(t, i, lw=3, label='Infective')
    ax[1].set_title('Infectious Population')
    if e is not None: ax[1].plot(t, e, lw=3, label='Exposed')
    ax[1].set_ylim(0, 0.5)
    ax[1].set_xlabel('Time /days')
    ax[1].set_ylabel('Population (in 20,000)')

    ax[2].plot(s, i, lw=3, label='s, i trajectory')
    ax[2].plot([1 / R0_seir, 1 / R0_seir], [0, 1], '--', lw=3, label='di/dt = 0')
    ax[2].plot(s[0], i[0], '.', ms=20, label='Initial Condition')
    ax[2].plot(s[-1], i[-1], '.', ms=20, label='Final Condition')
    ax[2].set_title('State Trajectory')
    ax[2].set_aspect('equal')
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlim(0, 1.05)
    ax[2].set_xlabel('Susceptible (in 20,000)')
    ax[2].set_ylabel('Infectious (in 20,000)')

    for a in ax:
        a.grid(True)
        a.legend()

    plt.tight_layout()


# SIR Model implementation result
t = np.linspace(0, 122, 122)
x_init_sir = s_init_sir, i_init_sir, r_init_sir
sol_sir = odeint(deriv, x_init_sir, t, args=(beta_sir, gamma_sir))
s, i, r = sol_sir.T
e = None
plotdata_sir(t, s, i)
plt.show()

# SEIR Model implementation result
t = np.linspace(0, 122, 122)
x_init_seir = s_init_seir, e_init_seir, i_init_seir, r_init_seir
sol_seir = odeint(seir_deriv, x_init_seir, t, args=(alpha_seir, beta_seir, gamma_seir))
s, e, i, r = sol_seir.T
plotdata_seir(t, s, i, e)
plt.show()

# Initialization for calculating R-squared for SIR and SEIR model
sir_inf_list = []
seir_inf_list = []
sir_rec_list = []
seir_rec_list = []
sir_inf_SSE = 0
seir_inf_SSE = 0
sir_rec_SSE = 0
seir_rec_SSE = 0

# Calculating SSE, RMSE, SST and R-squared for SIR and SEIR model
for i in range(0, len(sol_sir) - 20):
    sir_inf_list.append(int(sol_sir[i][1] * 20000))
    seir_inf_list.append(int(sol_seir[i][2] * 20000))
    sir_rec_list.append(int(sol_sir[i][-1] * 2000))
    seir_rec_list.append(int(sol_seir[i][-1] * 2000))
    sir_inf_SSE += (net_infect[i] - sir_inf_list[i]) ** 2
    seir_inf_SSE += (net_infect[i] - seir_inf_list[i]) ** 2
    sir_rec_SSE += (daily_recovery[i] - sir_rec_list[i]) ** 2
    seir_rec_SSE += (daily_recovery[i] - seir_rec_list[i]) ** 2

sir_inf_MSE = np.sqrt(sir_inf_SSE) / len(sol_sir)
seir_inf_MSE = np.sqrt(seir_inf_SSE) / len(sol_seir)
sir_rec_MSE = np.sqrt(sir_rec_SSE) / len(sol_sir)
seir_rec_MSE = np.sqrt(seir_rec_SSE) / len(sol_seir)
inf_SST = np.var(cum_infect) * len(sol_sir)
rec_SST = np.var(cum_recovery) * len(sol_sir)
sir_inf_R2 = 1 - sir_inf_SSE/inf_SST
seir_inf_R2 = 1 - seir_inf_SSE/inf_SST
sir_rec_R2 = 1 - sir_rec_SSE/rec_SST
seir_rec_R2 = 1 - seir_rec_SSE/rec_SST


# Comparing performance measure between SIR and SEIR model for infection
print('Evaluation measures of the SIR and SEIR model for cumulative infection')
print('RMSE of SIR model for infection:', sir_inf_MSE)
print('RMSE of SEIR model for infection:', seir_inf_MSE)
print('R-squared of SIR model for infection:', sir_inf_R2)
print('R-squared of SEIR model for infection:', seir_inf_R2)
print()

# Comparing performance measure between SIR and SEIR model for recovery
print('Evaluation measures of the SIR and SEIR model for cumulative recovery')
print('RMSE of SIR model for recovery:', sir_rec_MSE)
print('RMSE of SEIR model for recovery:', seir_rec_MSE)
print('R-squared of SIR model for recovery:', sir_rec_R2)
print('R-squared of SEIR model for recovery:', seir_rec_R2)


index = np.arange(2)
# Bar chart for MSE comparison between SIR and SEIR model
sirMSEs = (sir_inf_MSE, sir_rec_MSE)
seirMSEs = (seir_inf_MSE, seir_rec_MSE)
p1 = plt.bar(index-0.2, sirMSEs, width, color='navajowhite')
p2 = plt.bar(index+0.2, seirMSEs, width, color='lightskyblue')
plt.ylabel('Mean Squared Error')
plt.title('MSE comparison between SIR and SEIR')
plt.xticks(index, ('Infection', 'Recovery'))
plt.legend((p1[0], p2[0]), ('SIR', 'SEIR'))
plt.show()

# Bar chart for R-squared comparison between SIR and SEIR model
sirR2 = (sir_inf_R2, sir_rec_R2)
seirR2 = (seir_inf_R2, seir_rec_R2)
p1 = plt.bar(index-0.2, sirR2, width, color='aquamarine')
p2 = plt.bar(index+0.2, seirR2, width, color='salmon')
plt.ylabel('R-Squared')
plt.title('R-Squared comparison between SIR and SEIR')
plt.xticks(index, ('Infection', 'Recovery'))
plt.legend((p1[0], p2[0]), ('SIR', 'SEIR'))
plt.show()
