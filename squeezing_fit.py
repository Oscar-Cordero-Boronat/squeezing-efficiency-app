import streamlit as st
from typing import Union, List
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

class SqEfficiency:
    def __init__(self, 
                 power: np.ndarray, 
                 sq_data: np.ndarray, 
                 asq_data: np.ndarray, 
                 phase_noise: bool = False, 
                 detection_frequency: float = 5, 
                 decay_rate_cavity: float = 20.3,
                 y_axis = np.array([-3,15])
                                            ):
        """
        Initializes the SqEfficiency object with input data for power, squeezing, and antisqueezing values.
        
        Args:
            power (np.ndarray): Array of pump power values (mW).
            sq_data (np.ndarray): Array of squeezing data (dB).
            asq_data (np.ndarray): Array of antisqueezing data (dB).
            phase_noise (bool): Flag to indicate if phase noise should be considered in fitting.
            detection_frequency (float): Detection frequency (MHz).
            decay_rate_cavity (float): Decay rate of the cavity (MHz).
        """
        self.power = self._validate_input(power, np.ndarray, 'Power')
        self.sq_data = self._validate_input(sq_data, np.ndarray, 'Squeezing Data')
        self.asq_data = self._validate_input(asq_data, np.ndarray, 'Antisqueezing Data')
        self.phase_noise = self._validate_boolean(phase_noise, 'Phase Noise')
        self.omega = self._validate_float(detection_frequency, 'Detection Frequency', min_value=0)
        self.gamma = self._validate_float(decay_rate_cavity, 'Decay Rate Cavity', min_value=0)
        self.y_axis = y_axis

        if (self.phase_noise and len(self.sq_data) + len(self.asq_data) < 3):
            raise KeyError(f"It is required at least 2 squeezing and antisqueezing data") 
        elif  len(self.sq_data) != len(self.asq_data) or len(self.sq_data) != len(self.power):
            raise KeyError(f"You should have the same number of points in squeezing data, antisqueezing data and pump power") 
        elif (not self.phase_noise and len(self.sq_data) + len(self.asq_data) < 1):
            raise KeyError(f"It is required at least 1 squeezing and antisqueezing data") 
        else:
            # Fit the curve and plot the results
            self.fit_curve_noise()

    def _validate_input(self, data: Union[np.ndarray, List], expected_type: type, name: str) -> np.ndarray:
        """
        Validate that input data is of the expected type and has the correct shape.

        Args:
            data (Union[np.ndarray, List]): The input data to be validated.
            expected_type (type): The expected type of the input data.
            name (str): Name of the data being validated, used for error messages.

        Returns:
            np.ndarray: The validated input data.
        """
        if not isinstance(data, expected_type):
            raise ValueError(f"{name} must be of type {expected_type}.")
        
        # Ensure data is a numpy array for consistency
        if isinstance(data, list):
            data = np.array(data)
        
        return data

    def _validate_boolean(self, value: bool, name: str) -> bool:
        """
        Validate that the value is a boolean.

        Args:
            value (bool): The value to validate.
            name (str): The name used for error reporting.

        Returns:
            bool: The validated boolean value.
        """
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be of type bool.")
        return value

    def _validate_float(self, value: float, name: str, min_value: float = None, max_value: float = None) -> float:
        """
        Validate that the value is a float and within the specified range.

        Args:
            value (float): The value to validate.
            name (str): The name used for error reporting.
            min_value (float, optional): Minimum allowed value for the parameter.
            max_value (float, optional): Maximum allowed value for the parameter.

        Returns:
            float: The validated float value.
        """
        if not isinstance(value, (float, int)):
            raise ValueError(f"{name} must be of type float.")
        value = float(value)  # Convert to float if it's an integer
        
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be greater than or equal to {min_value}.")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be less than or equal to {max_value}.")
        
        return value

    def fit_curve_noise(self):
        """
        Fits the model to the squeezing and antisqueezing data using curve fitting.

        Updates the fitted parameters: eta, P_th, and phase_noise.
        """
        initial_guess = [0.8, 50, 0.2] if self.phase_noise else [0.8, 50]
        param_bounds = ([0, 0, 0], [1, np.inf, np.pi / 4]) if self.phase_noise else ([0, 0], [1, np.inf])
        
        # Fit model to both sq and asq data
        result = curve_fit(
            lambda p, eta, P_th, phase_noise=None: self.combined_residuals_noise((eta, P_th, phase_noise), p)
            if self.phase_noise else self.combined_residuals_noise_no_phase((eta, P_th), p),
            self.power,
            np.zeros(2 * len(self.power)),
            p0=initial_guess,
            bounds=param_bounds
        )

        # Extract fitted parameters
        if self.phase_noise:
            self.eta_fit, self.P_th_fit, self.phase_noise_fit = result[0]
        else:
            self.eta_fit, self.P_th_fit = result[0]
            self.phase_noise_fit = 0  # Set phase noise to 0 if not fitting it

        # Generate fitted curves for plotting
        self.power_fit = np.linspace(0, self.P_th_fit, 1000)
        self.sq_fit, self.asq_fit = self.sq_asq_model_noise(self.power_fit, self.eta_fit, self.P_th_fit, self.phase_noise_fit)

    

    def sq_asq_model_noise(self, power: np.ndarray, eta: float, P_th: float, phase_noise: float) -> tuple:
        """
        Model for squeezing and antisqueezing based on power, eta, P_th, and phase noise.

        Args:
            power (np.ndarray): Array of power values.
            eta (float): Squeezing efficiency.
            P_th (float): Threshold power.
            phase_noise (float): Phase noise (if any).

        Returns:
            tuple: Squeezing and antisqueezing values (dB).
        """
        c = np.sqrt(power / P_th)
        sq = 1 - eta * (4 * c) / ((1 + c)**2 + (self.omega / self.gamma)**2)
        asq = 1 + eta * (4 * c) / ((1 - c)**2 + (self.omega / self.gamma)**2)

        if self.phase_noise:
            sq, asq = 10 * np.log10(sq * np.cos(phase_noise)**2 + asq * np.sin(phase_noise)**2), \
                      10 * np.log10(asq * np.cos(phase_noise)**2 + sq * np.sin(phase_noise)**2)
        else:
            sq = 10 * np.log10(sq)
            asq = 10 * np.log10(asq)

        return sq, asq

    def combined_residuals_noise(self, params: tuple, power: np.ndarray) -> np.ndarray:
        """
        Computes the residuals (difference between the data and the model) for fitting.
        
        Args:
            params (tuple): Parameters (eta, P_th, phase_noise) to be fitted.
            power (np.ndarray): Array of power values.

        Returns:
            np.ndarray: Residuals for fitting.
        """
        eta, P_th, phase_noise = params
        sq, asq = self.sq_asq_model_noise(power, eta, P_th, phase_noise)
        residuals = np.concatenate((sq - self.sq_data, asq - self.asq_data))
        return residuals

    def combined_residuals_noise_no_phase(self, params: tuple, power: np.ndarray) -> np.ndarray:
        """
        Computes the residuals without considering phase noise.

        Args:
            params (tuple): Parameters (eta, P_th) to be fitted.
            power (np.ndarray): Array of power values.

        Returns:
            np.ndarray: Residuals for fitting.
        """
        eta, P_th = params
        sq, asq = self.sq_asq_model_noise(power, eta, P_th, 0)  # No phase noise
        residuals = np.concatenate((sq - self.sq_data, asq - self.asq_data))
        return residuals
    

    def plot_noise(self):
        fig, ax = plt.subplots()
        ax.plot(self.power, self.sq_data, 'or', label="Squeezing Data")
        ax.plot(self.power, self.asq_data, 'ob', label="Antisqueezing Data")
        ax.plot(self.power_fit, self.sq_fit, 'r-', label="Squeezing Fit")
        ax.plot(self.power_fit, self.asq_fit, 'b-', label="Antisqueezing Fit")
        ax.set_xlabel("Power [mW]")
        ax.set_ylabel("Variance [dB]")
        ax.set_xlim([0, self.P_th_fit])
        ymax = self.y_axis[1] // 2.5
        ymin = self.y_axis[0] // 2.5
        ax.set_yticks(np.linspace(ymin * 2.5, ymax * 2.5, int(ymax - ymin) + 1))
        ax.set_ylim(self.y_axis)
        ax.legend()
        ax.set_title(rf"$\eta = {self.eta_fit * 100:.2f}\text{{\%}}~~~~P_\text{{th}} = {self.P_th_fit:.2f}\,\text{{mW}}~~~~\varepsilon = {self.phase_noise_fit * 1e3:.2f}\,\text{{mrad}}$")
        ax.grid()
        return fig


class Gain:
    def __init__(self, pump_power, V, V0, y_axis = [0,50]):
        self.P = np.array(pump_power)
        self.V = np.array(V)
        self.V0 = np.array(V0)
        self.G = self.V / self.V0
        self.log_G = np.log(self.G)
        self.y_axis = y_axis

        if len(self.V) != len(self.V0) or len(self.V) != len(self.P):
            raise KeyError("Pump power, α², and α₀² should have the same number of points")
        else:
            # Fit the curve and plot the results
            self.fit_Pth()


    def gain_function(self, P, P_th):
        return 1 / (1 - (P / P_th)**.5) ** 2
    

    def fit_function(self, P, P_th):
        return np.log(1 / (1 - (P / P_th)**.5) ** 2)
    

    def fit_Pth(self, initial_guess=40):
        # Fit the threshold power P_th
        params, _ = curve_fit(self.fit_function, self.P, self.log_G, p0=[initial_guess])
        self.P_th_fitted = params[0]
    



    def plot_fit(self):
        self.P_fit = np.linspace(0, self.P_th_fitted, 500)
        self.V_fit = self.gain_function(self.P_fit, self.P_th_fitted)

        fig, ax = plt.subplots()
        ax.plot(self.P, self.G, 'or', label="Gain Data")
        ax.plot(self.P_fit, self.V_fit, 'r-', label="Gain Fit")

        ax.set_xlabel("Power [mW]")
        ax.set_ylabel("Gain")        
        ax.set_ylim(self.y_axis)
        ax.set_xlim([0, self.P_th_fitted])
        ax.grid()
        ax.legend()
        return fig
    
    def plot_fit_log(self):
        self.P_fit = np.linspace(0, self.P_th_fitted, 5000)
        self.V_fit = self.gain_function(self.P_fit, self.P_th_fitted)

        fig, ax = plt.subplots()
        ax.semilogy(self.P, self.G, 'or', label="Gain Data")
        ax.semilogy(self.P_fit, self.V_fit, 'r-', label="Gain Fit")

        ax.set_xlabel("Power [mW]")
        ax.set_ylabel("Gain")        
        ax.set_ylim(self.y_axis)
        ax.set_xlim([0, self.P_th_fitted])
        ax.grid()
        ax.legend()
        return fig



# Create a multipage app using Streamlit
st.set_page_config(page_title="Lab App", page_icon=":chart_with_upwards_trend:")

# Define the app's main pages
def squeezing_efficiency_analysis():
    st.title("Squeezing Efficiency Analysis")

    st.write(r"""
    This app analyzes the squeezing efficiency of a system by fitting the input squeezing and antisqueezing data into a theoretical model. 
    You can input the pump power, squeezing, antisqueezing data to visualize the fit and calculate key parameters such as squeezing efficiency, threshold power, and optional phase noise. 
    Decay rate and detection frequency can also be considered according to the theoretical model.
    """)

    # Display the equation using LaTeX
    st.latex(r"SQ = 1 - \eta\cdot\frac{4\sqrt{\frac{P}{P_{th}}}}{\bigg(1 + \sqrt{\frac{P}{P_{th}}}\bigg)^2 + \bigg(\frac{f}{f_{HWHM}}\bigg)^2}")
    st.latex(r"ASQ = 1 + \eta\cdot\frac{4\sqrt{\frac{P}{P_{th}}}}{\bigg(1 - \sqrt{\frac{P}{P_{th}}}\bigg)^2 + \bigg(\frac{f}{f_{HWHM}}\bigg)^2}")
    st.latex(r"VAR(SQ) = 10 \cdot \log_{10}\Big(SQ \cdot \cos(\varepsilon)^2 + ASQ \cdot \sin(\varepsilon)^2\Big)")
    st.latex(r"VAR(ASQ) = 10 \cdot \log_{10}\Big(ASQ \cdot \cos(\varepsilon)^2 + SQ \cdot \sin(\varepsilon)^2\Big)")

    # Inputs
    st.sidebar.header("Input Parameters")
    power = st.sidebar.text_input("Pump Power [mW] (comma-separated)", "6,10")
    sq_data = st.sidebar.text_input("Squeezing Data [dB] (comma-separated)", "-1.5,-2")
    asq_data = st.sidebar.text_input("Antisqueezing Data [dB] (comma-separated)", "4,6")
    phase_noise = st.sidebar.checkbox(" Include Phase Noise?", value=False)
    detection_frequency = st.sidebar.text_input(r"Detection Frequency $f$ [MHz]", "5")
    decay_rate_cavity = st.sidebar.text_input(r"Decay Rate Cavity $f_{HWHM}$ [MHz]", "20.3")
    y_axis = st.sidebar.text_input("y-axis limits (comma-separated)", "-3,15")

    # Convert inputs
    power = np.array([float(x) for x in power.split(",")])
    sq_data = np.array([float(x) for x in sq_data.split(",")])
    asq_data = np.array([float(x) for x in asq_data.split(",")])
    detection_frequency = float(detection_frequency)
    decay_rate_cavity = float(decay_rate_cavity)
    y_axis = np.array([float(x) for x in y_axis.split(",")])

    # Run the analysis
    if st.sidebar.button("Analyze"):
        try:
            # Perform analysis
            analysis = SqEfficiency(power, sq_data, asq_data, phase_noise=phase_noise, detection_frequency=detection_frequency, decay_rate_cavity=decay_rate_cavity, y_axis=y_axis)
            fig = analysis.plot_noise()

            # Display the plot
            st.pyplot(fig)

            # Add save/download button
            st.write("Click below to download the figure:")
            buf = io.BytesIO()  # Create an in-memory buffer
            fig.savefig(buf, format="png", dpi=500)  # Save the figure into the buffer
            buf.seek(0)  # Rewind the buffer to the beginning
            st.download_button(
                label="Download Figure",
                data=buf,
                file_name="squeezing_efficiency_plot.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Error: {e}")

def intracavity():
    st.title("Intra-Cavity Loss")

    st.write(r"""
        This app calculates the intra-cavity loss given 4 parameters. There are two possible expressions, the first one $\mathcal{L}_1$ describes the full expression without any approximation
        . The second one $\mathcal{L}_2$ is only valid for really low transmissions $T << 1$
        """)
    
    st.latex(r"\mathcal{L}_1 = T \cdot \frac{1 - \frac{P_{\text{refl}}-(1-m)P_{\text{in}}}{mP_{\text{in}}}}{\Bigg(1+\sqrt{\frac{P_{\text{refl}} - (1-m)P_{\text{in}}}{mP_{\text{in}}}\cdot(1-T)}\Bigg)^2}")
    st.latex(r"\mathcal{L}_2 = T \cdot \frac{1 - \sqrt{\frac{P_{\text{refl}} - (1-m)P_{\text{in}}}{mP_{\text{in}}}}}{1 + \sqrt{\frac{P_{\text{refl}} - (1-m)P_{\text{in}}}{mP_{\text{in}}}}}")
    
    T = st.text_input(r"Cavity transmission mirror $T$ [0-1]", "0.055")
    T = float(T)
    P_refl = st.text_input(r"Reflected Power at resonance $P_{refl}$", "1.02")
    P_refl = float(P_refl)
    P_in = st.text_input(r"Reflected Power at resonance $P_{in}$", "1.07")
    P_in = float(P_in)
    m = st.text_input(r"Mode matching $m$ [0-1]", "0.98")
    m = float(m)

    # Error checks
    valid_input = True

    if not (0 <= m <= 1):
        st.error("Mode matching ($m$) must be between 0 and 1 (exclusive).")
        valid_input = False
    if not (0 <= T <= 1):
        st.error("Cavity transmission mirror ($T$) must be between 0 and 1 (exclusive).")
        valid_input = False
    if not (P_in >= P_refl):
        st.error("Reflected power out of resonance ($P_{in}$) must be greater than reflected power at resonance ($P_{refl}$).")
        valid_input = False
    
    if valid_input:

        numerator_inside = P_refl - (1 - m) * P_in
        denominator_inside = m * P_in
        intermediate_fraction = numerator_inside / denominator_inside

        # Calculate the final formula
        L1 = T * (1 - intermediate_fraction) / ((1 + np.sqrt(intermediate_fraction * (1 - T))) ** 2)

        intermediate_step = np.sqrt((P_refl - (1-m)*P_in)/(m* P_in))

        L2 = T * (1 - intermediate_step)/(1 + intermediate_step)



        st.latex(rf"\text{{Intra-cavity loss: }} \mathcal{{L}}_1 = {L1 * 100:.2f}\%")
        st.latex(rf"\text{{Intra-cavity loss: }} \mathcal{{L}}_2 = {L2 * 100:.2f}\%")



def GainFit():
    st.title("Gain Analysis")
    st.write("""
        This app fits the gain function to the data. The gain can be expressed as a function
        of the fields and pump power, however the measured values in the scope are fields amplitude
        squared. Gain can be expressed as:""")
             
    st.latex(r"\Big(\frac{\alpha}{\alpha_0} \Big)^2 = \Bigg(\frac{1}{1 - \sqrt{P/P_{th}}} \Bigg)^2")
      
    st.write("""
             However, if we want the uncertainty of each point of the fitting to have a constant errorbar (e.g. 10%
             of the pump power, then it is better to fit the logarithm of this function:
             """)
    

    st.latex(r"\ln\Big(\frac{\alpha}{\alpha_0} \Big)^2 = 2 \cdot \ln\Bigg(\frac{1}{1 - \sqrt{P/P_{th}}} \Bigg)")
    st.sidebar.header("Input Parameters")
    P = st.sidebar.text_input("Pump Power [mW] (comma-separated)", "6, 10")
    P = np.array([float(x) for x in P.split(",")])

    V = st.sidebar.text_input(r"$\alpha^2$ (comma-separated)", "2, 4")
    V = np.array([float(x) for x in V.split(",")])

    V0 = st.sidebar.text_input(r"$\alpha_0^2$ (comma-separated)", "1, 1")
    V0 = np.array([float(x) for x in V0.split(",")])

    y_axis = st.sidebar.text_input("y-axis limits (comma-separated)", "1,30")
    y_axis = np.array([float(x) for x in y_axis.split(",")])

    log_scale = st.sidebar.checkbox("Log Scale?", value=False)



    # Run the analysis
    if st.sidebar.button("Analyze"):
        try:
            analysis = Gain(pump_power = P, V = V, V0 = V0, y_axis=y_axis)

            if not log_scale:
                fig = analysis.plot_fit()
            else:
                fig = analysis.plot_fit_log()

            # Display the plot
            st.pyplot(fig)
            st.latex(rf"P_{{th}} = {analysis.P_th_fitted:.2f}\text{{ mW}}")
                        
            # Add save/download button
            st.write("Click below to download the figure:")
            buf = io.BytesIO()  # Create an in-memory buffer
            fig.savefig(buf, format="png", dpi=500)  # Save the figure into the buffer
            buf.seek(0)  # Rewind the buffer to the beginning
            st.download_button(
                label="Download Figure",
                data=buf,
                file_name="gain_plot.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error: {e}")




# Define the navigation menu
menu = ["Squeezing Efficiency", "Intra-Cavity Loss", "Gain"]
choice = st.sidebar.selectbox("Select a Page:", menu)

# Page routing
if choice == "Squeezing Efficiency":
    squeezing_efficiency_analysis()
elif choice == "Intra-Cavity Loss":
    intracavity()
elif choice == "Gain":
    GainFit()
