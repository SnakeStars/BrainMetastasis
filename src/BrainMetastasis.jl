module BrainMetastasis
    using DataFrames
    using Dates
    using CSV
    using Statistics
    # using Plots, StatsPlots
    using MLJ
    using Logging
    using StatisticalMeasures
    using StatisticalMeasuresBase
    using Shapley, CategoricalDistributions
    import Imbalance

    function dataRename!(df::DataFrame)
        rename!(df, [:Sex,
                     :CountOfFractions,
                     :KarnovskiyIndex,
                     :CountOfGM,
                     :SummVol,
                     :Progression,
                     :RMZ,
                     :NMRL,
                     :Melanoma,
                     :RP,
                     :KRR,
                     :OVGM,
                     :OperationGM,
                     :ChemistryTherapy,
                     :TargetTherapy,
                     :NoMedic,
                     :AgeOnCRT,
                     :ReactionTime,
                     :MetastasisTime])
    end

    data = CSV.read("completed.csv", DataFrame)
    dataRename!(data)

    function PreparingBestModels()
        # X = DataFrames.select(data, Not(:"Интракраниальная прогрессия"))  # Все колонки, кроме 'target'
        # Y = DataFrames.select(data,"Интракраниальная прогрессия")              # Целевая переменная

        models = []

        y, X = unpack(data, ==(:Progression), rng=1234)
        y = coerce(y, OrderedFactor)
        model = (@load KNNClassifier verbosity=0)()
        K_range = range(model, :K, lower=5, upper=20)
        self_tuning_knn = TunedModel(
            model=model,
            resampling = StratifiedCV(nfolds=10, rng=1234),
            tuning = Grid(),
            range = K_range,
            measure = FScore(; beta=2),
            operation = predict_mode
        )
        mach = machine(self_tuning_knn, X, y)
        fit!(mach, verbosity=0)
        push!(models, self_tuning_knn)

        tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
        m_depth_range = range(tree, :max_depth, lower=-1, upper=80);
        self_tuning_tree = TunedModel(
            model=tree,
            resampling = StratifiedCV(nfolds=10, rng=1234),
            tuning = Grid(),
            range = m_depth_range,
            measure = FScore(; beta=2),
            operation = predict_mode
        )
        mach = machine(self_tuning_tree, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)

        Forest = (@load RandomForestClassifier pkg=DecisionTree verbosity=0)()
        m_depth_range = range(Forest, :max_depth, lower=-1, upper=80)
        n_sub_range = range(Forest, :n_subfeatures, lower=-1, upper=18)
        n_tree_range = range(Forest, :n_trees, lower=2, upper=100)
        self_tuning_forest = TunedModel(
            model=Forest,
            resampling = StratifiedCV(nfolds=10, rng=1234),
            tuning = RandomSearch(),
            range = [m_depth_range, n_sub_range, n_tree_range],
            measure = FScore(; beta=2),
            operation = predict_mode
        )
        mach = machine(self_tuning_forest, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)

        SVC = (@load ProbabilisticSVC pkg=LIBSVM verbosity=0)()
        mach = machine(SVC, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)

        LogClass = (@load LogisticClassifier pkg=MLJLinearModels verbosity=0)()
        mach = machine(LogClass, X, y)
        fit!(mach, verbosity=0)
        push!(models, mach)
    end
end # module BrainMetastasis
