using DataFrames
using Dates
using CSV
using MLJ
using Statistics
using Plots

function ReadFile()
    table = CSV.read("../train.csv", DataFrame)
end

function IntervalLength(s)
    Q1 = quantile(s, 0.25)
    Q2 = quantile(s, 0.5)
    Q3 = quantile(s, 0.75)
    arr = []
    for i in s
        for j in s
            if (i <= Q2) && (j >= Q2)
                if i == j == Q2
                    push!(arr, 0.0)
                else
                    push!(arr, ((j - Q2) - (Q2 - i))/(j - i))
                end
            end
        end
    end

    MC = median(arr)
    if MC >= 0
        maximum1 = Q3 + 1.5 * exp(3 * MC) * (Q3 - Q1)
        minimum1 = Q1 - 1.5 * exp((-4) * MC) * (Q3 - Q1)
    else
        maximum1 = Q3 + 1.5 * exp(4 * MC) * (Q3 - Q1)
        minimum1 = Q1 - 1.5 * exp((-3) * MC) * (Q3 - Q1)
    end
    maxarr = maximum(s[s .<= maximum1])
    minarr = minimum(s[s .>= minimum1])

    if minarr == maxarr
        maxarr = quantile(s, 0.95)
        minarr = quantile(s, 0.05)
    end
    if minarr == maxarr
        maxarr = maximum(s)
        minarr = minimum(s)
    end
    adjusted_scale_value = maxarr - minarr
    if adjusted_scale_value == 0.0
        adjusted_scale_value = 1
    end
    return adjusted_scale_value
end

function NormalizingData(table)
    cols = ["Число фракций СРТ", "Индекс Карновского", "Число очагов в ГМ", "Суммарный объем очагов", "Возраст на момент СРТ", "Время реагирования", "Время метастазирования"]
    mapcols!(s -> s .- median(s),table,cols=cols) # centering
    mapcols!(s -> s ./ IntervalLength(s), table, cols=cols)
    
    return table
end

function PreparingData()
    table = CSV.read("../train_changed.csv", DataFrame)

    function MakeData(str)
        if str !== missing && match(r"\d\d.\d\d.\d\d\d\d", str) !== nothing
            return Date(str, dateformat"dd.mm.yyyy")
        end
        return str
    end

    # Удаление миссингов
    filter!(row -> row."Интракраниальная прогрессия" !== missing, table::AbstractDataFrame)[!, ["Интракраниальная прогрессия"]]
    filter!(row -> row."Дата постановки онкологического диагноза / начала первичного лечения" !== missing, table::AbstractDataFrame)
    filter!(row -> (row."Дата развития МГМ" !== missing) && (match(r"\d\d.\d\d.\d\d\d\d",row."Дата развития МГМ") !== nothing), table::AbstractDataFrame)
    
    # Преобразуем даты
    transform!(table, "Дата постановки онкологического диагноза / начала первичного лечения" => (s -> MakeData.(s))
    => "Дата постановки онкологического диагноза / начала первичного лечения")
    transform!(table, "Дата рождения" => (s -> MakeData.(s))
    => "Дата рождения")
    transform!(table, "Дата удаления первичного очага" => (s -> MakeData.(s))
    => "Дата удаления первичного очага")
    transform!(table, "Дата развития МГМ" => (s -> MakeData.(s))
    => "Дата развития МГМ")
    transform!(table, "Дата проведения ОВГМ" => (s -> MakeData.(s))
    => "Дата проведения ОВГМ")
    transform!(table, "Дата операции на ГМ" => (s -> MakeData.(s))
    => "Дата операции на ГМ")
    transform!(table, "Дата 1-ой СРТ" => (s -> MakeData.(s))
    => "Дата 1-ой СРТ")

    

    transform!(table, "Пол" => (s -> ifelse.(uppercase.(s) .== "Ж", 0, 1)) => "Пол")

    transform!(table, "Онкологический диагноз" => (s -> ifelse.(s .== "РМЖ", 1, 0)) => "РМЖ")[!, ["РМЖ"]]
    transform!(table, "Онкологический диагноз" => (s -> ifelse.(s .== "НМРЛ", 1, 0)) => "НМРЛ")[!, ["НМРЛ"]]
    transform!(table, "Онкологический диагноз" => (s -> ifelse.(s .== "Меланома", 1, 0)) => "Меланома")[!, ["Меланома"]]
    transform!(table, "Онкологический диагноз" => (s -> ifelse.(s .== "РП", 1, 0)) => "РП")[!, ["РП"]]
    transform!(table, "Онкологический диагноз" => (s -> ifelse.(s .== "КРР", 1, 0)) => "КРР")[!, ["КРР"]]

    transform!(table, ["Дистантные метастазы", "Локальный рецидив"] => ((f,s) -> ifelse.(((f .=== missing) .|| (f .== "нет")) .&& ((s .=== missing) .|| (s .== "нет")), 0, 1)) 
    => "Интракраниальная прогрессия")

    select!(table, Not(["Дата удаления первичного очага", "Активирующие мутации", "Экстракраниальные метастазы", "Дистантные метастазы", "Локальный рецидив"]))
    transform!(table, "Дата проведения ОВГМ" => (s -> ifelse.((s .=== missing) .|| (s .== "нет"), 0, 1)) => "ОВГМ")
    transform!(table, "Дата операции на ГМ" => (s -> ifelse.((s .=== missing) .|| (s .== "нет"), 0, 1)) => "Операция на ГМ")[!, ["Дата развития МГМ"]]
    
    filter!(row -> (row."Дата 1-ой СРТ" - row."Дата развития МГМ") >= Day(0), table)[!, ["Дата развития МГМ", "Дата 1-ой СРТ"]]

    transform!(table, "Лекарственное лечение" => (s -> ifelse.((s .!== missing) .&& (s .== "Химиотерапия"), 1, 0)) => "Химиотерапия")[!, ["РМЖ"]]
    transform!(table, "Лекарственное лечение" => (s -> ifelse.((s .!== missing) .&& (s .== "Таргетная терапия"), 1, 0)) => "Таргетная терапия")[!, ["НМРЛ"]]
    transform!(table, "Лекарственное лечение" => (s -> ifelse.((s .!== missing) .&& (s .!= "Таргетная терапия") .&& (s .!= "Химиотерапия"), 1, 0)) => "Без лечения")[!, ["Химиотерапия", "Таргетная терапия", "Без лечения"]]

    transform!(table, ["Дата 1-ой СРТ","Дата рождения"]=> ((s, f) -> Dates.value.(s .- f)) => "Возраст на момент СРТ")[!, ["Возраст на момент СРТ"]]
    transform!(table, ["Дата 1-ой СРТ","Дата развития МГМ"]=> ((s, f) -> Dates.value.(s .- f)) => "Время реагирования")[!, ["Время реагирования"]]
    transform!(table, ["Дата развития МГМ","Дата постановки онкологического диагноза / начала первичного лечения"]=> ((s, f) -> Dates.value.(s .- f)) => "Время метастазирования")[!, ["Возраст на момент СРТ","Время реагирования","Время метастазирования"]]

    select!(table, Not(["Дата проведения ОВГМ","Дата операции на ГМ","Дата рождения","Онкологический диагноз","Дата постановки онкологического диагноза / начала первичного лечения", "Дата развития МГМ", "Дата 1-ой СРТ", "Объем максимального очага", "Лекарственное лечение"]))
    
    transform!(table, ["Суммарный объем очагов"] => (s -> parse.(Float64, replace.(s, "," => "."))) => "Суммарный объем очагов")

    function correlationMatrix(df::DataFrame)
        M = cor(Matrix(df))
        cols = names(df)
        (n,m) = size(M)
        heatmap(M, fc=cgrad(:seismic), xticks=(1:m,cols), clim=(-1,1),
                xrot=90, yticks=(1:m,cols), yflip=true, size=(1200, 1200))
        annotate!([(j, i, text(round(M[i,j],digits=3), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])
    end

    # savefig(correlationMatrix(table), "corrMatrix.png")

    # CSV.write("completed.csv", table)

    NormalizingData(table)

end