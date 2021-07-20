import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from lifelines import KaplanMeierFitter
from tensorflow.keras.models import Model, load_model
import tensorflow as tf

SEED = 7


def auc_helper(labels, preds, include_ci=False, n_bootstraps=1000):
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    roc_auc = metrics.auc(fpr, tpr)

    if include_ci:
        rng = np.random.default_rng(SEED)
        bs_aucs = []

        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(labels), len(labels))
            if len(np.unique(labels[indices])) < 2:
                continue

            auc_score = metrics.roc_auc_score(labels[indices], preds[indices])
            bs_aucs.append(auc_score)
        bs_aucs = np.array(bs_aucs)
        bs_aucs.sort()
        ci_lower = bs_aucs[int(0.025 * len(bs_aucs))]
        ci_upper = bs_aucs[int(0.975 * len(bs_aucs))]
        return fpr, tpr, roc_auc, ci_lower, ci_upper
    return fpr, tpr, roc_auc


def auc_plotter(fpr, tpr, auc, mdl_label=None):
    assert type(fpr) == type(tpr) == type(auc)

    plt.plot(fpr, tpr, label=mdl_label + ' AUC: %0.2f' % auc)
    plt.title('Model ROC Scores')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_kl_curve(time_to_event, labels, plot_label):
    surv_mdl = KaplanMeierFitter()
    surv_mdl.fit(durations=time_to_event, event_observed=labels)
    surv_mdl.plot(ci_alpha=0, label=plot_label)

    plt.title("Kaplan-Meier curve")
    plt.xlabel("Days after last scan")
    plt.ylabel("Survival")
    plt.show()


def cf_nri_helper(y_true, base_mdl, new_mdl):
    event_idx = np.where(y_true == 1.)[0]
    total_events = event_idx.shape[0]
    nonevent_idx = np.where(y_true == 0)[0]
    total_nonevents = event_idx.shape[0]

    # positive change = higher risk, negative_change = lower risk
    change_direction = new_mdl - base_mdl

    # events where change was in right direction (= risk went up)
    events_pos = np.where(change_direction[event_idx] > 0)[0].shape[0]
    # events where change was in wrong direciton (= risk lowered)
    events_neg = np.where(change_direction[event_idx] < 0)[0].shape[0]
    cfnri_events = (events_pos / total_events) - (events_neg / total_events)

    # non-events where change was in right direction (= risk lowered)
    nonevents_pos = np.where(change_direction[nonevent_idx] < 0)[0].shape[0]
    # non-events where change was in wrong direciton (= risk went up)
    nonevents_neg = np.where(change_direction[nonevent_idx] > 0)[0].shape[0]
    cfnri_nonevents = (nonevents_pos / total_nonevents) - (nonevents_neg /
                                                           total_nonevents)

    return (cfnri_events + cfnri_nonevents)


def bootstrap_results(y_truth, y_pred, num_bootstraps=1000):
    n_bootstraps = num_bootstraps
    rng_seed = 7  # control reproducibility
    y_pred = y_pred
    y_true = y_truth
    rng = np.random.RandomState(rng_seed)
    tprs = []
    fprs = []
    aucs = []
    threshs = []
    base_thresh = np.linspace(0, 1, 101)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        fpr, tpr, thresh = metrics.roc_curve(y_true[indices], y_pred[indices])
        thresh = thresh[1:]
        thresh = np.append(thresh, [0.0])
        thresh = thresh[::-1]
        fpr = np.interp(base_thresh, thresh, fpr[::-1])
        tpr = np.interp(base_thresh, thresh, tpr[::-1])
        tprs.append(tpr)
        fprs.append(fpr)
        threshs.append(thresh)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    fprs = np.array(fprs)
    mean_fprs = fprs.mean(axis=0)

    return mean_fprs, mean_tprs, base_thresh


def area_between_curves(y1, y2):
    diff = y1 - y2  # calculate difference
    posPart = np.maximum(diff, 0)
    negPart = -np.minimum(diff, 0)
    posArea = np.trapz(posPart)
    negArea = np.trapz(negPart)
    return posArea, negArea, posArea - negArea


def plot_idi(y_truth, ref_model, new_model, save=False):
    mean_fprs, mean_tprs, base = bootstrap_results(y_truth, new_model, 100)
    mean_fprs2, mean_tprs2, base2 = bootstrap_results(y_truth, ref_model, 100)
    is_pos, is_neg, idi_event = area_between_curves(mean_tprs, mean_tprs2)
    ip_pos, ip_neg, idi_nonevent = area_between_curves(mean_fprs2, mean_fprs)
    cf_nri = cf_nri_helper(y_truth, ref_model, new_model)
    print('IS positive', round(is_pos, 2), 'IS negative', round(is_neg, 2),
          'IDI events', round(idi_event, 2))
    print('IP positive', round(ip_pos, 2), 'IP negative', round(ip_neg, 2),
          'IDI nonevents', round(idi_nonevent, 2))
    print('IDI =', round(idi_event + idi_nonevent, 2))
    print(f'cfNRI = {round(cf_nri,2)}')
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    lw = 2
    plt.plot(base,
             mean_tprs,
             'black',
             alpha=0.5,
             label='Deaths combined modality')
    plt.plot(base,
             mean_fprs,
             'red',
             alpha=0.5,
             label='Non-deaths combined modality')
    plt.plot(base2,
             mean_tprs2,
             'black',
             alpha=0.7,
             linestyle='--',
             label='Deaths metadata')
    plt.plot(base2,
             mean_fprs2,
             'red',
             alpha=0.7,
             linestyle='--',
             label='Non-deaths metadata')
    plt.fill_between(base,
                     mean_tprs,
                     mean_tprs2,
                     color='black',
                     alpha=0.1,
                     label='Integrated sensitivity (area = %0.2f)' % idi_event)
    plt.fill_between(base,
                     mean_fprs,
                     mean_fprs2,
                     color='red',
                     alpha=0.1,
                     label='Integrated specificity (area = %0.2f)' %
                     idi_nonevent)

    plt.xlim([0.0, 1.10])
    plt.ylim([0.0, 1.10])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Calculated risk', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Sensitivity (black), 1 - specificity (red)', fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc="upper right", fontsize=11)
    plt.legend(loc=0, fontsize=11)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Metadata vs. combined modality sequence model IDI', fontsize=15)
    if save is not None:
        plt.savefig(f'./figures/{save}_idi_plot.png',
                    dpi=100,
                    bbox_inches='tight')
    plt.show()


def _interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def _compute_gradients(nn_mdl, images):
    with tf.GradientTape() as tape:
        tape.watch(images)
        prob = nn_mdl(images)
    return tape.gradient(prob, images)


def _integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(image, nn_mdl, m_steps=50, batch_size=16):
    im_tensor = tf.convert_to_tensor(np.squeeze(image), tf.float32)
    baseline = tf.zeros(shape=im_tensor.shape, dtype=tf.float32)

    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Initialize TensorArray outside loop to collect gradients.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = _interpolate_images(baseline=baseline,
                                                            image=image,
                                                            alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = _compute_gradients(
            nn_mdl=nn_mdl, images=interpolated_path_input_batch)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to),
                                                    gradient_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = _integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    int_grad_results = (image - baseline) * avg_gradients

    return int_grad_results