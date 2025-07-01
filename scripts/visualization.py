import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

def plot_pickups_by_hour(df, save_path=None, show=False):
    plt.figure(figsize=(10,6))
    sns.countplot(x='Hour', data=df, palette='viridis')
    plt.title('Uber Pickups by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Pickups')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_pickups_by_dayofweek(df, save_path=None, show=False):
    plt.figure(figsize=(10,6))
    sns.countplot(x='DayOfWeek', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], palette='magma')
    plt.title('Uber Pickups by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Pickups')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_heatmap_day_hour(df, save_path=None, show=False):
    pivot = df.pivot_table(index='DayOfWeek', columns='Hour', values='Date/Time', aggfunc='count').reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    plt.figure(figsize=(14,6))
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title('Heatmap of Uber Pickups by Day and Hour')
    plt.xlabel('Hour')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
