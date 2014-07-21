__author__ = 'davidnola'
import sklearn
from sklearn import *
from sklearn import ensemble
import MultilayerPerceptron

models_new_short = [
                [MultilayerPerceptron.MultilayerPerceptronManager ,{}                                       ],

                [sklearn.svm.SVC, {}    ],
                [sklearn.svm.SVC, {'C':.03}    ],
                [sklearn.svm.SVC, {'C':.003}    ],

                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .03}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .3}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .003}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .0003}    ],



                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .03}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .3}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .003}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .0003}  ],



                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .03}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .003}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .3}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .0003}  ],

                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':10}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':100}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':50}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':25}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':5}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':150}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':15}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':75}],


                [sklearn.naive_bayes.GaussianNB , {}],


                [sklearn.naive_bayes.BernoulliNB , {'alpha' : .5}],



                [sklearn.tree.DecisionTreeClassifier, {}],
                [sklearn.tree.DecisionTreeClassifier, {'max_features' : 'auto'}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 4}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_leaf' : 2}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_leaf' : 3}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 4, 'min_samples_leaf' : 3}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 2, 'min_samples_leaf' : 5}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 2, 'min_samples_leaf' : 3}],


                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 5}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 3}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 10}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 25}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 15}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 35}],


                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .1, 'n_estimators' : 100}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .3, 'n_estimators' : 100}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .1, 'n_estimators' : 400}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .05, 'n_estimators' : 200}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .01, 'n_estimators' : 500}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .5, 'n_estimators' : 50}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .1, 'n_estimators' : 40}],

                ]#END

models_kitchen_sink = [


                [sklearn.svm.SVC, {}    ],
                [sklearn.svm.SVC, {'C':.3}    ],
                [sklearn.svm.SVC, {'C':.03}    ],
                [sklearn.svm.SVC, {'C':.003}    ],

                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .03}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .3}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': 3}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .003}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .0003}    ],
                [sklearn.svm.LinearSVC, {'penalty' : 'l2', 'C': .000003}    ],


                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .03}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .3}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': 3}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .003}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .0003}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l2', 'C': .000003}  ],


                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .03}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .003}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .3}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': 3}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .0003}  ],
                [linear_model.LogisticRegression, {'penalty' : 'l1', 'C': .000003}  ],

                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':10}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':100}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':50}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':25}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':5}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':150}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':15}],
                [sklearn.ensemble.RandomForestClassifier, {'n_estimators':75}],


                [sklearn.naive_bayes.GaussianNB , {}],


                [sklearn.naive_bayes.BernoulliNB , {'alpha' : 1}],
                [sklearn.naive_bayes.BernoulliNB , {'alpha' : .5}],
                [sklearn.naive_bayes.BernoulliNB , {'alpha' : .25}],
                [sklearn.naive_bayes.BernoulliNB , {'alpha' : 0}],

                [sklearn.cross_decomposition.PLSRegression, {'n_components':2}],
                [sklearn.cross_decomposition.PLSRegression, {'n_components':5}],
                [sklearn.cross_decomposition.PLSRegression, {'n_components':10}],
                [sklearn.cross_decomposition.PLSRegression, {'n_components':3}],
                [sklearn.cross_decomposition.PLSRegression, {'n_components':12}],

                # #[sklearn.cross_decomposition.PLSSVD, {}],


                [sklearn.tree.DecisionTreeClassifier, {}],
                [sklearn.tree.DecisionTreeClassifier, {'max_features' : 'auto'}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 4}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_leaf' : 2}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_leaf' : 3}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 4, 'min_samples_leaf' : 3}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 5, 'min_samples_leaf' : 4}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 2, 'min_samples_leaf' : 5}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 3, 'min_samples_leaf' : 1}],
                [sklearn.tree.DecisionTreeClassifier, {'min_samples_split' : 2, 'min_samples_leaf' : 3}],


                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 5}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 3}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 10}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 25}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 15}],
                [sklearn.neighbors.KNeighborsClassifier, {'n_neighbors' : 35}],


                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .1, 'n_estimators' : 100}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .3, 'n_estimators' : 100}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .1, 'n_estimators' : 400}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .05, 'n_estimators' : 200}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .01, 'n_estimators' : 500}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .5, 'n_estimators' : 50}],
                [sklearn.ensemble.GradientBoostingClassifier, {'learning_rate' : .1, 'n_estimators' : 40}],

                ]#END





                #with everything
                # [0.9745944092115526, 0.9822526476828437, 0.97887736745305609, 0.98602799496411753, 0.98602271009555154, 0.98389315126083143, 0.97782973713761812, 0.97394678790045675]
                # 0.980430600713
                #Expanded: 0.97924800187

                # Bottom disabled
                #[0.94581387821569241, 0.94602939300126621, 0.92908981498192966, 0.95457019809695509, 0.93044958475115835, 0.93713977732415132, 0.92200584039683353, 0.94571786386523315]
                # 0.938852043829
                # All but best:
                #[0.97776464647234451, 0.96150179669861635, 0.96587931095133561, 0.96590112881633206, 0.96726244934351169, 0.96964795371451096, 0.97567353944155011, 0.95855775101356389]
                # 0.967773572056
                #


models_best = [
            [sklearn.linear_model.LogisticRegression ,{'penalty': 'l2', 'C': 3}                                 ],
            [sklearn.tree.DecisionTreeClassifier ,{'min_samples_split': 5, 'min_samples_leaf': 4}       ],
            [sklearn.linear_model.LogisticRegression ,{'penalty': 'l2', 'C': 0.3}                               ],
            [sklearn.neighbors.KNeighborsClassifier ,{'n_neighbors': 10}                                     ],
            [sklearn.ensemble.GradientBoostingClassifier ,{'n_estimators': 100, 'learning_rate': 0.1}       ],
            [sklearn.svm.LinearSVC ,{'penalty': 'l2', 'C': 0.3}                                        ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 150}                                 ],
            [sklearn.svm.LinearSVC ,{'penalty': 'l2', 'C': 0.0003}                                     ],
            [sklearn.svm.LinearSVC ,{'penalty': 'l2', 'C': 3}                                          ],
            [sklearn.linear_model.LogisticRegression ,{'penalty': 'l1', 'C': 0.3}                               ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 5}                                   ],
            [sklearn.ensemble.GradientBoostingClassifier ,{'n_estimators': 200, 'learning_rate': 0.05}      ],
            [sklearn.tree.DecisionTreeClassifier ,{'max_features': 'auto'}                              ],
            [sklearn.tree.DecisionTreeClassifier ,{'min_samples_leaf': 2}                               ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 100}                                 ],
            [sklearn.svm.LinearSVC ,{'penalty': 'l2', 'C': 0.003}                                      ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 10}                                  ],
            [sklearn.ensemble.GradientBoostingClassifier ,{'n_estimators': 400, 'learning_rate': 0.1}       ],
            [sklearn.linear_model.LogisticRegression ,{'penalty': 'l1', 'C': 3}                                 ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 25}                                  ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 75}                                  ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 50}                                  ],
            [sklearn.svm.LinearSVC ,{'penalty': 'l2', 'C': 0.03}                                       ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 15}                                  ],
            [sklearn.ensemble.GradientBoostingClassifier ,{'n_estimators': 100, 'learning_rate': 0.3}       ],
            [sklearn.neighbors.KNeighborsClassifier ,{'n_neighbors': 5}                                      ],
            [sklearn.ensemble.GradientBoostingClassifier ,{'n_estimators': 50, 'learning_rate': 0.5}        ],
            [sklearn.neighbors.KNeighborsClassifier ,{'n_neighbors': 3}                                      ],
        ]

models_small = [
                    [sklearn.svm.LinearSVC ,{'penalty': 'l2', 'C': 0.03}                                       ],
            [sklearn.ensemble.RandomForestClassifier ,{'n_estimators': 15}                                  ],
            [sklearn.ensemble.GradientBoostingClassifier ,{'n_estimators': 100, 'learning_rate': 0.3}       ],
            [sklearn.neighbors.KNeighborsClassifier ,{'n_neighbors': 5}                                      ],
            [sklearn.ensemble.GradientBoostingClassifier ,{'n_estimators': 50, 'learning_rate': 0.5}        ],
            [sklearn.neighbors.KNeighborsClassifier ,{'n_neighbors': 3}                                      ],

                    ]

models_MLP = [
                    [MultilayerPerceptron.MultilayerPerceptronManager ,{}                                       ],

                    ]