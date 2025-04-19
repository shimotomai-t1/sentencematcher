#smatch.py

from bert_score import score
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import numpy
import numpy as np



# prompt: ２つのクラスタの確率密度値が一致する点を２分法で求めたい。初期値は0と1からはじめてください

def f(x, means, covariances):
  return (1 / (covariances[0] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - means[0]) / covariances[0])**2) - (1 / (covariances[1] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - means[1]) / covariances[1])**2)

def bisection(a, b, tolerance, means, covariances):
    if f(a,means,covariances) * f(b,means,covariances) >= 0:
        print("Bisection method fails.")
        return None

    while (b - a) / 2 > tolerance:
        c = (a + b) / 2
        if f(c, means, covariances) == 0:
            return c
        elif f(a, means,covariances) * f(c,means,covariances) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# prompt: simの値を元にDPマッチングをしてください。縦横方向のスコアはすでに求めたrootの値をつかい、斜め方向のスコアはsimの値を使い、経路の合計スコアが最大となる経路を探してください。
def dp_matching(sim, root):
  n = len(sim)
  m = len(sim[0])

  # Initialize the DP table
  dp = np.zeros((n + 1, m + 1))

  # Fill the DP table
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      # Calculate scores for horizontal, vertical, and diagonal movements
      horizontal_score = dp[i - 1, j] + root
      vertical_score = dp[i, j - 1] + root
      diagonal_score = dp[i - 1, j - 1] + sim[i-1][j-1]

      # Choose the maximum score
      dp[i, j] = max(horizontal_score, vertical_score, diagonal_score)

  # Backtrack to find the optimal path
  path = []
  i = n
  j = m
  while i > 0 or j > 0:
    path.append((i -1, j - 1))
    if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + sim[i - 1][j - 1]:
      i -= 1
      j -= 1
    elif i > 0 and dp[i,j] == dp[i-1, j] + root:
      i -= 1
    else:
      j -= 1

  return dp[n, m], path[::-1] # Return the maximum score and the optimal path

# prompt: 最適経路のoptimal_pathを図示してください。
def plot_optimal_path(sim, optimal_path):
    """Plots the similarity matrix with the optimal path highlighted."""
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    plt.imshow(sim, cmap='viridis', interpolation='nearest')  # Use a suitable colormap
    plt.colorbar(label='Similarity Score')

    # Extract x and y coordinates from the optimal path
    x_coords = [pair[1]+0.5 for pair in optimal_path]
    y_coords = [pair[0]+0.5 for pair in optimal_path]

    plt.plot(x_coords, y_coords, marker='o', color='red', linestyle='-')  # Plot the path

    plt.xlabel("English Sentences")
    plt.ylabel("Translated Sentences")
    plt.title("Optimal Path for Sentence Alignment")
    plt.xticks(range(len(sim[0])))  # Set x-axis ticks
    plt.yticks(range(len(sim)))    # Set y-axis ticks
    plt.show()



# prompt: A群B群の２群の整数リストがあり、要素をノードとしたときのエッジの情報がopathとして入力例のように与えられたときに、出力例のように接続関係のあるノード群をひとまとめにしたい。リストの要素のタプルはA群とB群の整数要素を表している。群の中のエッジは存在せずすべてA群とB群の間のエッジのみ存在するとして処理する関数を作ってください
# 入力例： opath=[(1, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 6)]
# 出力例：[[(1,1)],[(2,2), (2,3), (2,4), (2,5), (3,5), (4,5)],[(5,6)]

def aggregate_connections(opath):
    """
    Groups connected nodes from two groups based on edge information.

    Args:
        opath: A list of tuples, where each tuple represents an edge
               connecting a node from group A to a node from group B.

    Returns:
        A list of lists, where each sublist contains connected nodes.
    """

    groups = []
    node_map = {}  # Map nodes to their group index

    for a, b in opath:
        node = (a, b)
        found_group = False
        for i, group in enumerate(groups):
            if any((a, _) in group or (_, b) in group for _, _ in group):
              groups[i].append(node)
              node_map[node] = i
              found_group = True
              break
        if not found_group:
          groups.append([node])
          node_map[node] = len(groups) - 1
    return groups

def corresponder(A,B, connections):
  alist = [{a for a,b in c} for c in connections]
  print(alist)
  blist = [{b for a,b in c} for c in connections]
  print(blist)
  anew = [[A[i] for i in elm] for elm in alist]
  bnew = [[B[i] for i in elm] for elm in blist]
  return anew, bnew



def main():
    # prompt: sentenceBERTを使って同様にengtxtとtranslated_literalのそれぞれの文をすべての組み合わせで類似度を計算してください。
    # Load the Sentence Transformer model
    model = SentenceTransformer('all-mpnet-base-v2')
    # Compute sentence embeddings for all sentences
    engtxt_embeddings = model.encode(engtxt)
    translated_literal_embeddings = model.encode(translated_literal)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(engtxt_embeddings, translated_literal_embeddings)
    sim = model.similarity(engtxt_embeddings, translated_literal_embeddings)
    print(sim)
    plt.pcolor(sim)

    plt.pcolor(-numpy.log(numpy.array(sim)+0.0001))
    plt.colorbar()
    -numpy.log(numpy.array(sim)+0.0001)

    numpy.median(sim)

    plt.hist(sim.flatten(), bins=100)
    plt.show()

    # prompt: simを１次元の値に変換してから１次元GMM２クラスタでフィッティングしてください。結果推定されたパラメータを表示してください。２つのガウス分布をプロットしてください。
    # Assuming 'sim' is already defined from the previous code
    sim_1d = sim.flatten()  # Convert sim to a 1D array

    # Fit a 1D 2-component GMM
    gmm = GaussianMixture(n_components=2, random_state=0).fit(sim_1d.reshape(-1, 1))

    # Estimated parameters
    means = gmm.means_.flatten()
    covariances = np.sqrt(gmm.covariances_.flatten())  # Standard deviations
    weights = gmm.weights_

    print("Estimated means:", means)
    print("Estimated standard deviations:", covariances)
    print("Estimated weights:", weights)

    # Plot the two Gaussian distributions
    x = np.linspace(sim_1d.min(), sim_1d.max(), 100)
    y1 =  (1 / (covariances[0] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - means[0]) / covariances[0])**2)
    y2 =  (1 / (covariances[1] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - means[1]) / covariances[1])**2)


    plt.plot(x, y1, label='Gaussian 1')
    plt.plot(x, y2, label='Gaussian 2')
    plt.hist(sim_1d, bins=100, density=True, alpha=0.5, label='Data') # Overlay histogram
    plt.xlabel('Similarity Score')
    plt.ylabel('Probability Density')
    plt.title('2-Component GMM Fit')
    plt.legend()
    plt.show()

    # Example usage
    a = 0
    b = 1
    tolerance = 1e-6
    root = bisection(a, b, tolerance, means, covariances)

    #if root is not None:
    #    print(f"The root is approximately: {root}")
    #    print(f"f({root}) = {f(root)}")
    # Example root value (replace with your actual root value)
    #root = 0.5

    max_score, optimal_path = dp_matching(sim, root)
    print(f"Maximum score: {max_score}")
    print("Optimal path:", optimal_path)

    # Example usage (assuming 'sim' and 'optimal_path' are defined from the previous code)
    plot_optimal_path(sim, optimal_path)
    print(translated_literal, len(translated_literal))

    #opath=[(1, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 6)]
    #aggregate_connections(opath)
    aggregate_connections(optimal_path)
    result = corresponder(engtxt, translated_literal, aggregate_connections(optimal_path))

    print(result)


# prompt: 文章の比較をするので２０行程度の日本語の文を作ってください。内容は全体で一貫性のあるストーリーにして、１行は５文字から１００文字の幅を持ったものにしてください。文字列はリストとしてoriginalというインスタンスにしてください
original = [
    "静かな湖面に夕日が映る。",
    "水面には、かすかな波紋が広がる。",
    "一羽の白鳥が優雅に水面を滑る。",
    "その姿は、まるで絵画のようだった。",
    "湖畔には、古びた小さな小屋がある。",
    "小屋の窓から、温かい光が漏れている。",
    "そこには、老婆が静かに読書をしている。",
    "彼女は、長い人生の思い出に浸っていた。",
    "時折、湖面を見つめる彼女の視線は、遠くを見据えているようだった。",
    "湖の向こうには、山々が連なっている。",
    "夕焼けに染まった山々は、雄大な姿を見せている。",
    "空には、星が一つ、二つと輝き始める。",
    "静寂の中で、自然の音だけが聞こえる。",
    "風の音、水のせせらぎ、鳥のさえずり。",
    "老婆は、本を閉じて窓の外を眺める。",
    "静かな湖面と美しい夕焼けに心が安らぐ。",
    "彼女は、明日への希望を感じていた。",
    "穏やかな夜空の下、湖は静かに眠りにつく。",
    "白鳥は、遠くの空へと飛び立っていった。",
    "そして、湖畔の小屋にも、静かな夜が訪れる。"
]

# prompt: 上のoriginal全体で統合して、英語に翻訳してください。わかりやすいように文章間に意味を補う文章を入れてください。その後各文に分解してtranslatedというリストにしてください
# Combine the sentences to form a coherent paragraph.
combined_text = " ".join(original)

# Translate the combined text into English.
eng_text = "A tranquil lake reflects the setting sun.  On the water's surface, faint ripples spread. A single white swan glides gracefully across the water; its form is like a painting. By the lakeshore, there stands an old, small cottage. Warm light leaks from the cottage's window. Inside, an old woman quietly reads. She is immersed in the memories of her long life. Occasionally, her gaze, fixed upon the lake, seems to look far into the distance. Beyond the lake, mountains stretch out. The mountains, painted by the sunset, reveal their majestic appearance. In the sky, stars begin to shine, one by one. In the silence, only the sounds of nature are audible – the sound of the wind, the murmuring water, and the birdsong. The old woman closes her book and gazes out the window. Her heart finds peace in the tranquil lake and the beautiful sunset. She feels hope for tomorrow. Under the calm night sky, the lake quietly falls asleep. The white swan takes flight toward the distant sky. Then, a quiet night descends upon the lakeside cottage."
#Split the translated text into individual sentences.
engtxt = eng_text.split('. ')

# prompt: originalの各文を直訳してください。
translated_literal = [
    "Quiet lake surface, the setting sun is reflected.",
    "On the water surface, faint ripples spread.",
    "One white swan elegantly glides on the water surface.",
    "Its figure, it was just like a painting.",
    "On the lakeside, there is an old small hut.",
    "From the hut's window, warm light is leaking.",
    "There, an old woman is quietly reading.",
    "She was immersed in the memories of her long life.",
    "Occasionally, her gaze looking at the lake surface, it seemed to be looking far into the distance.",
    "Beyond the lake, mountains are lined up.",
    "Mountains dyed in the sunset, are showing a majestic figure.",
    "In the sky, stars begin to shine, one by one.",
    "In the silence, only the sound of nature is audible.",
    "The sound of the wind, the murmur of water, the chirping of birds.",
    "The old woman closes the book and looks out the window.",
    "The heart is soothed by the quiet lake surface and the beautiful sunset.",
    "She was feeling hope for tomorrow.",
    "Under the calm night sky, the lake quietly falls asleep.",
    "The white swan flew to the distant sky.",
    "And, a quiet night also comes to the lakeside hut."
]

if __name__ == '__main__':
    main()
    with open('translated.txt', 'w') as f:
        for line in translated_literal:
            f.write(line + '\n')
    with open('original.txt', 'w') as f:
        for line in original:
            f.write(line + '\n')
    