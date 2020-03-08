from argparse import ArgumentParser

from mir_eval.beat import evaluate

from beat_tracking_tcn.beat_tracker import predict_beats_from_spectrogram
from beat_tracking_tcn.datasets.ballroom_dataset import BallroomDataset

def parse_args():
    parser = ArgumentParser(
        description="Perform evaluation on a given BeatNet model")

    parser.add_argument("spectrogram_dir", type=str)
    parser.add_argument("label_dir", type=str)
    parser.add_argument("model_checkpoint", type=str)
    parser.add_argument("--downbeats", action="store_true")

    return parser.parse_args()

def evaluate_model(
        model_checkpoint,
        spectrogram,
        ground_truth,
        downbeats=False):

    prediction = predict_beats_from_spectrogram(
        spectrogram,
        model_checkpoint,
        downbeats=downbeats)

    if downbeats:
        scores = (evaluate(ground_truth[0], prediction[0]),
                  evaluate(ground_truth[1], prediction[1]))
    else:
        scores = evaluate(ground_truth, prediction)

    return scores

def evaluate_model_on_dataset(
        model_checkpoint,
        dataset,
        ground_truths,
        downbeats=False,
        print_callback=None):

    mean_scores = {}
    running_scores = {}
    for i in range(len(dataset)):
        spectrogram = dataset[i]["spectrogram"].unsqueeze(0)
        ground_truth = ground_truths[i]

        scores = evaluate_model(
            model_checkpoint,
            spectrogram,
            ground_truth,
            downbeats)

        if downbeats:
            beat_scores = scores[0]
            downbeat_scores = scores[1]
        else:
            beat_scores = scores

        for metric in beat_scores:
            if metric not in running_scores:
                running_scores[metric] = 0.0
            
            running_scores[metric] += beat_scores[metric]
        
        if print_callback is not None:
            print_callback(i, running_scores)

    for metric in running_scores:
        mean_scores[metric] = running_scores[metric] / (i + 1)        

    return {
        "total_examples": i + 1,
        "scores": mean_scores
    }

if __name__ == "__main__":
    args = parse_args()

    def print_callback(i, running_scores):
        def make_metric_heading(metric):
            words = metric.split(" ")
            for i, _ in enumerate(words):
                for vowel in "aeiouAEIOU":
                    words[i] = words[i][0] + words[i][1:].replace(vowel, "")
            return "".join(words)

        if i == 0:
            line = ""
            for metric in running_scores:
                metric_heading = make_metric_heading(metric)
                heading = " %s " % metric_heading
                if len(metric_heading) < 6:
                    padding_length = int((6 - len(metric_heading)) / 2)
                    print(padding_length)
                    padding = " " * padding_length
                    heading = padding + heading + padding
                    if len(metric_heading) % 2 == 1:
                        heading += " "
                heading += "|"
                line += heading
            print(line)

        line = ""
        for metric in running_scores:
            metric_heading = make_metric_heading(metric)
            number_length = len(metric_heading) - 2
            line += " {1:.{0}f} |".format(
                max(4, number_length),
                running_scores[metric] / (i + 1))
        print(line, end="\r")

    dataset = BallroomDataset(
        args.spectrogram_dir,
        args.label_dir,
        downbeats=args.downbeats)

    if args.downbeats:
        ground_truths = tuple(zip(
            [dataset.get_ground_truth(i) for i in range(len(dataset))],
            [dataset.get_ground_truth(i, downbeats=True)
                for i in range(len(dataset))]))
    else:
        ground_truths = (dataset.get_ground_truth(i) for i in range(len(dataset)))

    evaluate_model_on_dataset(
        args.model_checkpoint,
        dataset,
        ground_truths,
        args.downbeats,
        print_callback)