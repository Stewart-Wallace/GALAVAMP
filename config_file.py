learning_params = dict(
    ##########
    # Irreversible processes
	##########
	annihilation_rule 				= False,
	maximum_terms					= None,

	##########
	# Variance
	##########
	initial_variance 				= 0.1,
	variance_update_frequency		= None,
	variance_decrease				= 2,
	variance_increase				= 1.5,
    
	##########
	# Operator-Space Rates
	##########
    complexity						= 2,
	lindbladian_exponential_rate 	= 5,
	hamiltonian_exponential_rate	= 5,
    mutation_rate					= 0.5,
    random_rate						= 0.35,
    
	##########
	# Optimised Learning Parameters
	##########
	PARAMETER_chance		 		= 0.0,
    
    ADDINGH_chance 					= 0.25,
    REMOVINGH_chance				= 0.5,
    SWAPPINGH_chance 				= 0.15,
    
	ADDINGL_SINGLE_chance			= 0.25,
    REMOVINGL_SINGLE_chance 		= 0.5,
    SWAPPINGL_SINGLE_chance			= 0.15,
    
	NUTCRACKER_chance				= 0.25,
    
	term_cost						= 10,
    complicated_term_cost			= 30,    
    
	##########
	# Parameter Proposal Controls
	##########
	parameter_prior_shape 			= 0.03,
	parameter_prior_scale			= 20,
    
	parameter_proposal_shape 		= 2,										# 2.0		
	parameter_proposal_scale		= 0.3,										# 0.3
    
	##########
	# Easy Access
	##########
	project_name 					= 'PL1off_gene',
	cohort_size 					= 100,
    survival_ratio					= 0.5,
	mutation_steps					= 100,
)
	