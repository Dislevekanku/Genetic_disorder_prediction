def get_common_hyp_param_grid_multi(cap_x_df, m_points, fast_script_dev):
    if fast_script_dev:
        common_hyp_param_grid = {
            'preprocessor__numerical__imputer__strategy': ['mean'],  # default
            # 'preprocessor__nominal__target_encoder__smooth': ['auto']  # default
        }
    else:
        common_hyp_param_grid = {
            'preprocessor__numerical__imputer__strategy': ['mean', 'median'],  # default
            # 'preprocessor__nominal__target_encoder__smooth': ['auto']  # default
        }

    # Remove or modify the 'smooth' parameter in the PolynomialWrapper
    common_hyp_param_grid['preprocessor__nominal__target_encoder__feature_encoder__smooth'] = ['auto']

    return common_hyp_param_grid



def get_sgd_class_hyp_param_grid_multi(alpha_points, l1_ratio_points, m_points, cap_x_df, fast_script_dev):
    common_hyp_param_grid_multi = get_common_hyp_param_grid_multi(cap_x_df, m_points, fast_script_dev)

    if fast_script_dev:
        sgd_class_hyp_param_grid = {
            'estimator__penalty': ['l1', 'elasticnet'],
            'estimator__alpha': [0.0001, 0.001, 0.01],
            'estimator__l1_ratio': [0.1, 0.3, 0.5, 0.7],
            'estimator__n_jobs': [-1],
        }
    else:
        sgd_class_hyp_param_grid = {
            'estimator__penalty': ['l2'],
            'estimator__alpha': [0.0001],
            'estimator__l1_ratio': [0.15],
            'estimator__n_jobs': [None],
        }

    # Exclude 'smooth' when used in PolynomialWrapper
    sgd_class_hyp_param_grid = {key: value for key, value in sgd_class_hyp_param_grid.items() if 'smooth' not in key}

    sgd_class_hyp_param_grid = sgd_class_hyp_param_grid | common_hyp_param_grid_multi

    return sgd_class_hyp_param_grid


def get_dt_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev):
    common_hyp_param_grid_multi = get_common_hyp_param_grid_multi(cap_x_df, m_points, fast_script_dev)

    if fast_script_dev:
        dt_class_hyp_param_grid = {
            # tree growth hyperparameters
            # 'estimator__criterion': ['gini'],  # default
            # 'estimator__splitter': ['best'],  # default
            # 'estimator__max_depth': [None],  # default
            # 'estimator__max_features': [None],  # default
            # 'estimator__min_samples_split': [2],  # do not work with min_samples_split
            # 'estimator__min_samples_leaf': [1],  # do not work with min_samples_leaf
            # 'estimator__min_weight_fraction_leaf': [0.0],  # do not work with min_weight_fraction_leaf
            # 'estimator__random_state': [None],  # set at instantiation in notebook
            # 'estimator__max_leaf_nodes': [None],  # do not work with max_leaf_nodes
            # 'estimator__min_impurity_decrease': [0.0],  # do not work with min_impurity_decrease
            # 'estimator__class_weight': [None],  # set at instantiation in notebook
            # 'estimator__ccp_alpha': [0.0]  # do not work with pruning
        }
    else:
        dt_class_hyp_param_grid = {
            'estimator__criterion': ['gini', 'entropy'],  # Additional criterion options
            'estimator__splitter': ['best', 'random'],  # Additional splitter options
            'estimator__max_depth': [None, 10, 20],  # Additional max depth values
            'estimator__max_features': ['auto', 'sqrt', 'log2'],  # Additional max features options
            # 'estimator__min_samples_split': [2, 5, 10],  # Additional min samples split values
            # 'estimator__min_samples_leaf': [1, 2, 4],  # Additional min samples leaf values
            # 'estimator__min_weight_fraction_leaf': [0.0, 0.1],  # Additional min weight fraction leaf values
            # 'estimator__random_state': [42],  # Additional random state values
            # 'estimator__max_leaf_nodes': [None, 50, 100],  # Additional max leaf nodes values
            # 'estimator__min_impurity_decrease': [0.0, 0.1],  # Additional min impurity decrease values
            # 'estimator__class_weight': ['balanced', None],  # Additional class weight options
            # 'estimator__ccp_alpha': [0.0, 0.1],  # Additional ccp alpha values
        }

    dt_class_hyp_param_grid = dt_class_hyp_param_grid | common_hyp_param_grid_multi

    return dt_class_hyp_param_grid


def get_rf_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev):
    common_hyp_param_grid_multi = get_common_hyp_param_grid_multi(cap_x_df, m_points, fast_script_dev)

    if fast_script_dev:
        rf_class_hyp_param_grid = {
            # tree growth hyperparameters
            # 'estimator__criterion': ['gini'],  # default
            # 'estimator__max_depth': [None],  # default
            # 'estimator__max_features': ['sqrt'],  # default
            # 'estimator__min_samples_split': [2],  # do not work with min_samples_split
            # 'estimator__min_samples_leaf': [1],  # do not work with min_samples_leaf
            # 'estimator__min_weight_fraction_leaf': [0.0],  # do not work with min_weight_fraction_leaf
            # 'estimator__random_state': [None],  # set at instantiation in notebook
            # 'estimator__max_leaf_nodes': [None],  # do not work with max_leaf_nodes
            # 'estimator__min_impurity_decrease': [0.0],  # do not work with min_impurity_decrease
            # 'estimator__class_weight': [None],  # set at instantiation in notebook
            # 'estimator__ccp_alpha': [0.0]  # do not work with pruning
            # ensemble methods hyperparameters
            # 'estimator__n_estimators': [100],  # default
            # 'estimator__bootstrap': [True],  # default - do not work with bootstrap
            # 'estimator__oob_score': [False],  # default - do not work with oob_score
            # 'estimator__n_jobs': [-1],  # default=None
            # 'estimator__warm_start': [False],  # default - do not work with warm_start
            # 'estimator__max_samples': [None],  # default=None
            # other hyperparameters
            # 'estimator__verbose': [0],  # default - do not work with verbose
        }
    else:
        rf_class_hyp_param_grid = {
            # Tree growth hyperparameters
            'estimator__criterion': ['gini', 'entropy'],  # Additional criterion options
            'estimator__max_depth': [None, 10, 20],  # Additional max depth values
            'estimator__max_features': ['auto', 'sqrt', 'log2'],  # Additional max features options
            # 'estimator__min_samples_split': [2, 5, 10],  # Additional min samples split values
            # 'estimator__min_samples_leaf': [1, 2, 4],  # Additional min samples leaf values
            # 'estimator__min_weight_fraction_leaf': [0.0, 0.1],  # Additional min weight fraction leaf values
            # 'estimator__random_state': [42],  # Additional random state values
            # 'estimator__max_leaf_nodes': [None, 50, 100],  # Additional max leaf nodes values
            # 'estimator__min_impurity_decrease': [0.0, 0.1],  # Additional min impurity decrease values
            # 'estimator__class_weight': ['balanced', None],  # Additional class weight options
            # 'estimator__ccp_alpha': [0.0, 0.1],  # Additional ccp alpha values
            # Ensemble methods hyperparameters
            'estimator__n_estimators': [50, 150, 200],  # Additional n_estimators values
            # 'estimator__bootstrap': [True, False],  # Additional bootstrap options
            # 'estimator__oob_score': [True, False],  # Additional oob_score options
            'estimator__n_jobs': [-1],  # Default is None, -1 utilizes all available processors
            # 'estimator__warm_start': [True, False],  # Additional warm_start options
            'estimator__max_samples': [None, 0.5, 0.8],  # Additional max_samples values
            # Other hyperparameters
            # 'estimator__verbose': [0, 1],  # Additional verbose options
        }

    rf_class_hyp_param_grid = rf_class_hyp_param_grid | common_hyp_param_grid_multi

    return rf_class_hyp_param_grid


def get_ab_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev):
    common_hyp_param_grid_multi = get_common_hyp_param_grid_multi(cap_x_df, m_points, fast_script_dev)

    if fast_script_dev:
        ab_class_hyp_param_grid = {
        }
    else:
        ab_class_hyp_param_grid = {
        }

    ab_class_hyp_param_grid = ab_class_hyp_param_grid | common_hyp_param_grid_multi

    return ab_class_hyp_param_grid


def get_gb_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev):
    common_hyp_param_grid_multi = get_common_hyp_param_grid_multi(cap_x_df, m_points, fast_script_dev)

    if fast_script_dev:
        gb_class_hyp_param_grid = {
        }
    else:
        gb_class_hyp_param_grid = {
        }

    gb_class_hyp_param_grid = gb_class_hyp_param_grid | common_hyp_param_grid_multi

    return gb_class_hyp_param_grid


def get_hyp_param_grid_func_dict_multi(alpha_points, l1_ratio_points, m_points, cap_x_df, fast_script_dev):
    hyp_param_grid_func_dict = {
        'SGDClassifier': get_sgd_class_hyp_param_grid_multi(alpha_points, l1_ratio_points, m_points, cap_x_df,
                                                            fast_script_dev),
        'DecisionTreeClassifier': get_dt_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev),
        'RandomForestClassifier': get_rf_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev),
        'AdaBoostClassifier': get_ab_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev),
        'GradientBoostingClassifier': get_gb_class_hyp_param_grid_multi(m_points, cap_x_df, fast_script_dev)
    }
    return hyp_param_grid_func_dict


def get_hyp_param_tuning_exp_dict_multi(estimator_names, estimator_list, alpha_points, l1_ratio_points, m_points, cap_x_df,
                                        fast_script_dev=False, print_param_grids=False):
    hyp_param_grid_func_dict = \
        get_hyp_param_grid_func_dict_multi(alpha_points, l1_ratio_points, m_points, cap_x_df, fast_script_dev)

    param_grid_list = []
    for estimator in estimator_names:
        param_grid_list.append(hyp_param_grid_func_dict[estimator])

    hyp_param_tuning_exp_dict = dict(zip(estimator_list, param_grid_list))

    if print_param_grids:
        for estimator_name, estimator_param_grid in hyp_param_tuning_exp_dict.items():
            print(f'\n', '*' * 60, sep='')
            print('*' * 60, sep='')
            print(f'{estimator_name}\n{estimator_param_grid}', sep='')

    return hyp_param_tuning_exp_dict


if __name__ == '__main__':
    pass
