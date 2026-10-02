import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')


def plot_pickups_by_hour(df, save_path=None, show=False):
    hour_counts = df['Hour'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    hour_counts.plot(kind='bar', color='steelblue')
    plt.title('Uber Pickups by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Pickups')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_pickups_by_day(df, save_path=None, show=False):
    day_counts = df['Date/Time'].dt.day.value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    day_counts.plot(kind='line', marker='o', color='darkorange')
    plt.title('Uber Pickups by Day of Month')
    plt.xlabel('Day of Month')
    plt.ylabel('Number of Pickups')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_pickups_by_dayofweek(df, save_path=None, show=False):
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = df['DayOfWeekName'].value_counts().reindex(order)
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='crimson')
    plt.title('Uber Pickups by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Pickups')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_monthly_pickups(df, save_path=None, show=False):
    monthly_counts = df.groupby('Month').size()
    plt.figure(figsize=(8, 5))
    monthly_counts.plot(kind='line', marker='o', color='green')
    plt.title('Uber Pickups by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Pickups')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_heatmap_day_hour(df, save_path=None, show=False):
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = df.pivot_table(index='DayOfWeekName', columns='Hour', values='Date/Time', aggfunc='count').reindex(order)
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap='YlGnBu', cbar_kws={'label': 'Pickup Count'})
    plt.title('Heatmap of Uber Pickups by Day and Hour')
    plt.xlabel('Hour')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, save_path=None, show=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='royalblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--', linewidth=1)
    plt.title('Actual vs Predicted Pickup Count')
    plt.xlabel('Actual Pickup Count')
    plt.ylabel('Predicted Pickup Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_feature_importance(feature_names, importances, save_path=None, show=False):
    pairs = sorted(zip(feature_names, importances), key=lambda pair: pair[1], reverse=True)
    labels = [feature for feature, _ in pairs]
    scores = [score for _, score in pairs]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, scores, color='seagreen')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

