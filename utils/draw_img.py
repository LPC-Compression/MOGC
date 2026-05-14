from matplotlib import pyplot as plt
from matplotlib import rcParams

def _draw_metric_fig(MODELS_CONFIG, metric_key, title, ylabel, is_cd=False, legend_loc=4):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_title(title, fontsize=35)
    ax.grid(True)
    ax.set_xlabel('bpp', fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)

    for config in MODELS_CONFIG:
        name = config['name']
        data = config['data_var']
        color = config['color']
        
        x = [item['bpp'] for item in data]
        y = [item[metric_key] for item in data]
        if is_cd:
            y = [val * 100 for val in y]
        marker = 'o'

        ax.plot(x, y, '-', label=name, color=color, marker=marker, linewidth=5, markersize=15)

    ax.legend(loc=legend_loc, prop={'size': 35}, labelspacing=0.08, frameon=0.1)
    fig.tight_layout(pad=0.1)
    
    return fig


def draw_rd_curve(bpp_list, CD_list, D1_list, D2_list):
    from data.rd_data import MODELS_CONFIG
    ours = [{'bpp': bpp, 'cd': cd, 'PSNR_d1': d1, 'PSNR_d2': d2}\
             for bpp, cd, d1, d2 in zip(bpp_list, CD_list, D1_list, D2_list)]
    config = MODELS_CONFIG.copy()
    config.append({'name': 'Ours', 'data_var': ours, 'color': 'royalblue', 'marker': 'o'})

    d1_fig = _draw_metric_fig(
            config,
            metric_key='PSNR_d1', 
            title="PSNR D1 on SemanticKITTI", 
            ylabel='PSNR D1 (dB)', 
            legend_loc=4
        )
    d2_fig = _draw_metric_fig(
            config,
            metric_key='PSNR_d2', 
            title="PSNR D2 on SemanticKITTI", 
            ylabel='PSNR D2 (dB)', 
            legend_loc=4
        )
    cd_fig = _draw_metric_fig(
            config,
            metric_key='cd', 
            title="CD on SemanticKITTI", 
            ylabel='Chamfer Distance (cm)', 
            is_cd=True, 
            legend_loc=0
        )
    return d1_fig, d2_fig, cd_fig
