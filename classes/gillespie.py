import numpy as np

class Gillespie():
    def __init__(self, repression_threshold, hill_coefficient, mRNA_degradation_rate,
                 protein_degradation_rate, basal_transcription_rate, translation_rate,
                 transcription_delay):
        self.repression_threshold = repression_threshold
        self.hill_coefficient = hill_coefficient
        self.mRNA_degradation_rate = mRNA_degradation_rate
        self.protein_degradation_rate = protein_degradation_rate
        self.basal_transcription_rate = basal_transcription_rate
        self.translation_rate = translation_rate
        self.transcription_delay = transcription_delay
        return

    def gillespie_with_delay(self, N, initial_mRNA, initial_protein,
                            duration, number_of_time_samples = 500, equilibration_time = 0.0, number_of_trajectories = 1):

        '''Generate one trace of the Hes1 model. This function implements a stochastic version of
            the model model in 'Identifying oscillations using Gaussian processes' (2017).
            It applies the direct method described in 'Exact Stochastic Simulation of Coupled Chemical Reactions with Delay' (2007) as Algorithm 1.
            This method is an exact method to calculate the temporal evolution of stochastic reaction systems with delay.
            todo: need this ? At the end of the trajectory, transcription events will have been scheduled to occur after the trajectory has terminated. This function returns this transcription schedule, as well.

            Parameters
            ----------

            N : float
                N is system size as represented in the Hill Function for the
                System size - sets total number of particles in system.
            parameters : list
                parameters are, respectively
                    P0 = parameters[0]                          Appears in Hill Function, in the paper is called the constant representing the strength of negative repression
                    N_p = parameters[1]                         Appears in Hill Function, called in Jochen's code the hill_coefficient.
                    mRNA_degradation_rate = parameters[2]       Rate of mRNA Degradation ie mRNA_degradation_rate (MUM in original paper)
                    protein_degradation_rate = parameters[3]    Rate of Protein Degradation ie protein_degradation_rate (MUP in original paper)
                    translation_rate = parameters[4]            Rate of Protein Production Through Translation ie translation_rate parameter (ALPHAP in original paper)
                    transcription_rate = parameters[5]          Rate of mRNA Production Through Transcription ie transcription_rate (appears in Hill Function) (ALPHAM in original paper)
                    transcription_delay = parameters[6]         Transcription Delay, for a reaction triggered at t, an mRNA molecule is not produced until t + tau (tau in original paper)
            duration : float
                duration of the trace in hours.
            number_of_time_samples : float
                the number of time steps to be taken when between the interval starting from equilibration time and finishing at duration.
                The default is for the number of samples to be 500.
            equilibration_time : float
                add a neglected simulation period at beginning of the trajectory of length equilibration_time
                trajectory in order to get rid of any overshoots, for example.
                The default is for the equilibration_time to be 0.
            number_of_trajectories : integer
                decide how many traces you want to produce. The more traces, the higher the number of trajectories.
                The default is 1, any more will be significantly more time consuming.

            Returns
            -------
            sampling_times : ndarray

            mRNA_output : ndarray

            protein_output : ndarray

            '''

        P0 = self.repression_threshold                              # Appears in Hill Function, in the paper is called the constant representing the strength of negative repression, called repression_threshold in Jochen's code.
        N_p = self.hill_coefficient                                 # Appears in Hill Function, called in Jochen's code the hill_coefficient.
        mRNA_degradation_rate = self.mRNA_degradation_rate          # Rate of mRNA Degradation ie mRNA_degradation_rate (MUM in original paper)
        protein_degradation_rate = self.protein_degradation_rate    # Rate of Protein Degradation ie protein_degradation_rate (MUP in original paper)
        translation_rate = self.basal_transcription_rate            # Rate of Protein Production Through Translation ie translation_rate parameter (ALPHAP in original paper)
        transcription_rate = self.translation_rate                  # Rate of mRNA Production Through Transcription ie transcription_rate (appears in Hill Function) (ALPHAM in original paper)
        transcription_delay = self.transcription_delay              # Transcription Delay, for a reaction triggered at t, an mRNA molecule is not produced until t + tau (tau in original paper)

        total_time = duration + equilibration_time
        sampling_times = np.linspace(equilibration_time, total_time, number_of_time_samples)

        mRNA_output = np.zeros((number_of_trajectories, number_of_time_samples))       # Empty Array to store Outputs
        protein_output = np.zeros((number_of_trajectories, number_of_time_samples))    # Empty Array to store Outputs

        for i in range(0, number_of_trajectories):
            sampling_index = 0                      # Difference is that sampling_index keeps increasing for all time steps
            iter = 1                                # whilst iter increases only when a reaction happens.
            rlist = []                              # rlist is an empty list

        # Step 1. Initialise
            time = 0.0                                      # Setting start time to 0
            current_mRNA = initial_mRNA                     # Setting initial number of molecules
            current_protein = initial_protein               # Setting initial number of protein

        # Step 2. Generate Propensity Functions

            alpha_1 = mRNA_degradation_rate*current_mRNA                                      # mRNA Degradation
            alpha_2 = protein_degradation_rate*current_protein                                # Protein Degradation
            alpha_3 = translation_rate*current_mRNA                                           # Protein Production - Via Translation
            alpha_4 = N*transcription_rate/(1+((current_protein/float(N))/float(P0))**N_p)    # mRNA Production - Via Transcription - Using Hill function for Hes1 Protein. Also called repression
            # todo: this is how it was defined in original Matlab code, not really the same as the paper.
            # todo: Ask Jochen. In his code we have current protein/current_repression_threshold, claiming that P0 is the repression threshold
            # todo: Seems to me like in his code N which is system size does not exist. Whilst it very well does in the project's code and paper.
            # todo: is there an assumption made of N = 1?

        # Step 3. While loop to go through each time point

            while time < sampling_times[-1]:
                # Sum up propensities into obtaining alpha_0
                base_propensity = alpha_1 + alpha_2 + alpha_3 + alpha_4
                # Generate Random numbers r1 and r2 for Gillespie's SSA with Delay
                first_random_number, second_random_number = np.random.rand(2)
                # Generate Tau, ie time_to_next_reaction.
                time_to_next_reaction = (1/base_propensity)*np.log(1/first_random_number)

        # Step 4. If statement of whether delayed transcription times are being between t and t + dt

                    # mRNA Transcription
                if  len(rlist)> 0 and time <= rlist[0] and rlist[0] <= time + time_to_next_reaction: # delayed transcription execution
                                                                                                     # after being initiated with the last else: statement, it finally occurs.
                    current_mRNA += 1
                    current_time = rlist.pop(0)                                                      # Picks out the time from rlist and empties the list.

                    alpha_1 = current_mRNA*mRNA_degradation_rate                                     # Adjust the propensities based on the molecules 'used up' by the reaction.
                    alpha_3 = current_mRNA*translation_rate

        # Step 5. All statements within Else allow us to 'choose' the correct mu, hence answer to the question:
        #         WHICH reaction occurs at time + tau?
        #         and UPDATE # of molecules based on the reaction which occurred.

                else:
                    #Reaction 1 is mRNA Degradation
                    if second_random_number*base_propensity <= alpha_1:
                        current_mRNA -= 1
                        alpha_1 = mRNA_degradation_rate*current_mRNA
                        alpha_3 = translation_rate*current_mRNA

                    #Reaction 2 is Protein Degradation
                    elif alpha_1 <= second_random_number*base_propensity and second_random_number*base_propensity <= (alpha_1 + alpha_2):
                        current_protein -= 1
                        alpha_2 = protein_degradation_rate*current_mRNA
                        alpha_4 = N*transcription_rate/(1+((current_protein/float(N))/float(P0))**N_p)

                    #Reaction 3 is Protein Translation
                    elif (alpha_1 + alpha_2) <= second_random_number*base_propensity and second_random_number*base_propensity <= (alpha_1 + alpha_2 + alpha_3):
                        current_protein += 1
                        alpha_2 = protein_degradation_rate*current_protein
                        alpha_4 = N*transcription_rate/(1+((current_protein/float(N))/float(P0))**N_p)

                    #Reaction 4 is Transcription Initiation, means its ongoing but there's no visible result nor change to the mRNA count yet if delayed.
                    else:
                        if transcription_delay == 0:                                  # Meaning no-delay gillespie so transcription occurs.
                            current_mRNA += 1
                            alpha_1 = mRNA_degradation_rate*current_mRNA
                            alpha_3 = translation_rate*current_mRNA
                        else:
                            rlist.append(time + transcription_delay)                  # Meaning delay gillespie so transcription is initiated and ongoing.
                            #todo: what is happening here? Make sure you understand.

                    time += time_to_next_reaction

        # Step 6. Now we update the mRNA and Protein values based on the findings.

                iter += 1

        # Step 7. This while loop takes into account all the instances for which no reaction occurred and
        #         the mRNA and Protein levels stay the same, hence every input in the output up until the reaction will stay the same.

                while (sampling_index < len(sampling_times) and time > sampling_times[sampling_index]):
                    mRNA_output[i, sampling_index] = current_mRNA
                    protein_output[i, sampling_index] = current_protein
                    sampling_index += 1

            sampling_times -= equilibration_time


        return sampling_times, mRNA_output, protein_output

    # todo: need to add the power spectra analysis?? May not be necessary
