import pystim


bin_path = None  # TODO correct.
vec_path = None  # TODO correct.
trials_path = None  # TODO correct.
stimulus = pystim.stimuli.flashed_images.load(bin_path, vec_path, trials_path)

print(stimulus.nb_frames)
print(stimulus.nb_diplays)
print(stimulus.nb_trials)
print(stimulus.nb_conditions)
print(stimulus.condition_nbs)
print(stimulus.condition_nbs_sequence)
# print(stimulus.nb_repetitions)  # ill-defined?
print(stimulus.get_nb_repetitions(condition_nb))

print(stimulus.get_frame(display_nb))
print(stimulus.get_frame_by_display_nb(display_nb))

print(stimulus.get_nb_displays(trial_nb))
print(stimulus.get_display_nbs(trial_nb))
print(stimulus.get_nb_displays(condition_nb, condition_trial_nb))
print(stimulus.get_display_nbs(condition_nb, condition_trial_nb))

# Une condition c'est des paramètres et une (ou une suite) de binary frames.


# TODO stimulus doit permettre la génération.
# TODO stimulus doit permettre de vérifier son intégrité.
# TODO stimulus doit faciliter l'analyse.

stimulus.get_trial_display_extend(trial_nb)
stimulus.get_trial_display_extend(condition_nb, condition_trial_nb)

stimulus.get_trial_display_extends(condition_nb)

condition = stimulus.get_condition(condition_nb)  # une condition -> plusieurs trials, plusieurs displays
trial = stimulus.get_trial(trial_nb)  # un trial -> une condition, plusieurs displays
display = stimulus.get_display(display_nb)  # un display -> un trial, une condition

stimulus.get_display_nbs_extent(trial_nb)
stimulus.get_time_extent(trial_nb)

psr = response.get_peristimulus_responses(stimulus.get_trial_display_extends(condition_nb))


# Analyse.
# 1. Pour chaque enregistrement.
#   a. Visualizer le taux de décharge au cours temps (pour chaque neurone).