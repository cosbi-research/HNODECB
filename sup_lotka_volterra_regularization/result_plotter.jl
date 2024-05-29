#= 
Script to plot the results of the analysis
=#

cd(@__DIR__)

using Serialization, Plots, Statistics, LaTeXStrings, DataFrames

regularizer_df_e00 = deserialize("profiler_analysis_e0.00/regularizer_df.jld")
regularizer_df_e05 = deserialize("profiler_analysis_e0.05/regularizer_df.jld")

#mkdir plots
if !isdir("plots")
  mkdir("plots")
end


#sort by the mec parameter value 
sort!(regularizer_df_e00, :mec_parameter)
sort!(regularizer_df_e05, :mec_parameter)

gr()

plot_font = "Arial"
Plots.default(fontfamily=plot_font)

plt = Plots.plot(legend=:outerright, foreground_color_legend = nothing, xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, size=(850, 450), dpi=300, bottom_margin=20px, left_margin=20px)
scatter!(plt, regularizer_df_e00.mec_parameter, regularizer_df_e00.nn_contribution, color=:seagreen3, label=latexstring("DS_{0.00}"), markerstrokewidth=0, markersize=4)
plot!(plt, regularizer_df_e00.mec_parameter, regularizer_df_e00.nn_contribution, color=:seagreen3, label="")
scatter!(plt, regularizer_df_e05.mec_parameter, regularizer_df_e05.nn_contribution, color=:orange2, label=latexstring("DS_{0.05}"), markerstrokewidth=0, markersize=4)
plot!(plt, regularizer_df_e05.mec_parameter, regularizer_df_e05.nn_contribution, color=:orange2, label="")
vline!(plt, [1.3], color=:grey, linestyle=:dash, label="")
xaxis!(plt, "α", foreground_color_grid=:lightgrey)
yaxis!(plt, "Regularizer value", foreground_color_grid=:lightgrey)
Plots.svg(plt, "plots/lv_regularizer_analysis.svg")

#plots the MSE for the UDE models trained on DS_0.00
plt = Plots.plot(legend=false, xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, size=(750, 450), dpi=300, bottom_margin=20px, left_margin=20px)
scatter!(plt, regularizer_df_e00.mec_parameter, log10.(regularizer_df_e00.se_cost), color=:blue, markerstrokewidth=0, markersize=3, legend = false)
xaxis!("α", color=:grey, linestyle=:dash, label="")
yaxis!("MSE (log10)", color=:grey, linestyle=:dash, label="")
Plots.svg(plt, "plots/MSE_00.svg")

#correlation between the mechanistic parameter and the MSE
cor(regularizer_df_e00.mec_parameter, regularizer_df_e00.se_cost)

#plots the MSE for the UDE models trained on DS_0.00
plt = Plots.plot(legend=false, xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18,legendfontsize=18, size=(750, 450), dpi=300, bottom_margin=20px, left_margin=20px)
scatter!(plt, regularizer_df_e05.mec_parameter, log10.(regularizer_df_e05.se_cost), color=:blue, markerstrokewidth=0, markersize=3, legend = false)
xaxis!("α", color=:grey, linestyle=:dash, label="")
yticks!([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [" 0.0", " 0.5", " 1.0", " 1.5", " 2.0", " 2.5"])
yaxis!("MSE (log10)", color=:grey, linestyle=:dash, label="")
Plots.svg(plt, "plots/MSE_05.svg")

#correlation between the mechanistic parameter and the MSE
cor(regularizer_df_e05.mec_parameter, regularizer_df_e05.se_cost)