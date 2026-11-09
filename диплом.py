"""
Практична робота: маршрутизація групових заявок у мережі M/G/1.
3 вузли, 2 класи пріоритету; пріоритет неперервний (non-preemptive):
поточне обслуговування групи не переривається.


"""
import heapq
import random
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t

N, S = 3, 2
lam = np.array([0.6, 0.4])
mu = np.array([4.0, 3.2, 2.5])
cost = np.array([5.0, 2.0, 1.0])
sizes = np.array([1, 2, 3])
prob = np.array([0.5, 0.3, 0.2])
priority_weight = np.array([2.0, 1.0])
transport_weight = 0.02

E_size = np.sum(prob * sizes)
E_size2 = np.sum(prob * sizes**2)
ES = E_size / mu
# For a group of size B, service is Erlang(B, mu):
# E[S] = E[B]/mu; E[S^2] = (E[B^2] + E[B])/mu^2.
ES2 = (E_size2 + E_size) / mu**2


def matrix(x):
    return np.asarray(x).reshape(S, N)


def queue_metrics(U):
    """Mean queue waiting times Wq[node, class] and class-node loads."""
    Wq = np.zeros((N, S))
    rho = np.zeros((N, S))
    for i in range(N):
        rho[i, :] = lam * U[:, i] * ES[i]
        W0 = 0.5 * np.sum(lam * U[:, i] * ES2[i])
        for k in range(S):
            cumulative = np.sum(rho[i, :k + 1])
            higher = np.sum(rho[i, :k])
            denominator = (1.0 - cumulative) * (1.0 - higher)
            if denominator <= 0:
                return None, rho
            Wq[i, k] = W0 / denominator
    return Wq, rho


def objective(x):
    U = matrix(x)
    Wq, _ = queue_metrics(U)
    if Wq is None:
        return 1e9
    weighted_queue = sum(
        np.sum(priority_weight * lam * U[:, i] * Wq[i, :])
        for i in range(N)
    )
    transport = np.sum(lam[:, None] * U * cost[None, :])
    return weighted_queue + transport_weight * transport


constraints = [
    {"type": "eq", "fun": lambda x: matrix(x).sum(axis=1) - 1.0}
]
for i in range(N):
    constraints.append({
        "type": "ineq",
        "fun": lambda x, i=i:
            0.98 - np.sum(lam * matrix(x)[:, i] * ES[i])
    })

result = minimize(
    objective,
    np.full(S * N, 1.0 / N),
    method="SLSQP",
    bounds=[(0.0, 1.0)] * (S * N),
    constraints=constraints,
    options={"maxiter": 2000, "ftol": 1e-12}
)
if not result.success:
    raise RuntimeError(result.message)

U_opt = matrix(result.x)


def theoretical_total_time(U):
    Wq, _ = queue_metrics(U)
    times = []
    for k in range(S):
        rates = lam[k] * U[k, :]
        total_rate = np.sum(rates)
        times.append(np.sum(rates * (Wq[:, k] + ES)) / total_rate)
    return np.asarray(times)


def simulate(policy, U=None, horizon=30_000, warmup=3_000, seed=1):
    """
    Discrete-event simulation. Returns mean time-in-system by class.
    Jobs arriving after warmup and no later than horizon are measured.
    The simulation drains the system after the arrival horizon.
    """
    rng = random.Random(seed)
    queues = [[] for _ in range(N)]  # (class, sequence, arrival, group_size)
    busy = [False] * N
    finish_at = [0.0] * N
    completions = []  # (completion_time, node, class, arrival_time)
    completed = [[] for _ in range(S)]
    rr_counter = [0] * S
    t_now = 0.0
    sequence = 0
    total_lambda = float(np.sum(lam))

    while True:
        next_arrival = (
            t_now + rng.expovariate(total_lambda)
            if t_now < horizon else float("inf")
        )
        next_completion = completions[0][0] if completions else float("inf")

        if next_arrival <= next_completion and next_arrival <= horizon:
            t_now = next_arrival
            k = 0 if rng.random() < lam[0] / total_lambda else 1
            group_size = rng.choices(list(sizes), weights=list(prob))[0]

            if policy == "random":
                node = rng.randrange(N)
            elif policy == "equal":
                # Cyclic equal-count distribution, separately for each class.
                node = rr_counter[k] % N
                rr_counter[k] += 1
            elif policy == "shortest":
                counts = [
                    len(queues[i]) + (1 if busy[i] else 0)
                    for i in range(N)
                ]
                minimum = min(counts)
                candidates = [i for i, value in enumerate(counts)
                              if value == minimum]
                node = rng.choice(candidates)
            elif policy == "workload":
                workloads = []
                for i in range(N):
                    remaining = max(0.0, finish_at[i] - t_now) if busy[i] else 0.0
                    waiting_work = sum(
                        size / mu[i] for _, _, _, size in queues[i]
                    )
                    workloads.append(remaining + waiting_work)
                minimum = min(workloads)
                candidates = [i for i, value in enumerate(workloads)
                              if abs(value - minimum) < 1e-12]
                node = rng.choice(candidates)
            elif policy == "optimal":
                node = rng.choices(range(N), weights=list(U[k]))[0]
            else:
                raise ValueError("Unknown policy: " + str(policy))

            sequence += 1
            heapq.heappush(
                queues[node], (k, sequence, t_now, group_size)
            )

            if not busy[node]:
                kk, _, arrival_time, size = heapq.heappop(queues[node])
                service = sum(rng.expovariate(mu[node]) for _ in range(size))
                finish = t_now + service
                busy[node] = True
                finish_at[node] = finish
                heapq.heappush(
                    completions, (finish, node, kk, arrival_time)
                )

        elif next_completion < float("inf"):
            t_now, node, k, arrival_time = heapq.heappop(completions)
            if arrival_time >= warmup:
                completed[k].append(t_now - arrival_time)

            if queues[node]:
                kk, _, at, size = heapq.heappop(queues[node])
                service = sum(rng.expovariate(mu[node]) for _ in range(size))
                finish = t_now + service
                finish_at[node] = finish
                heapq.heappush(completions, (finish, node, kk, at))
            else:
                busy[node] = False
                finish_at[node] = 0.0
        else:
            break

    return np.array([np.mean(values) for values in completed])


def run_experiment(repetitions=30, horizon=30_000, warmup=3_000):
    policies = ["random", "equal", "shortest", "workload", "optimal"]
    all_results = {}
    for p_index, policy in enumerate(policies):
        samples = []
        for repetition in range(repetitions):
            samples.append(simulate(
                policy,
                U=U_opt,
                horizon=horizon,
                warmup=warmup,
                seed=1000 + repetition
            ))
        all_results[policy] = np.asarray(samples)

    print("\nОптимальна матриця U:")
    print(U_opt)
    print("Цільова функція:", objective(result.x))
    print("Завантаження вузлів:", queue_metrics(U_opt)[1].sum(axis=1))
    print("Аналітичний час у системі:", theoretical_total_time(U_opt))

    print("\nСередній час у системі та 95% ДІ:")
    for policy, values in all_results.items():
        aggregate = values @ lam / np.sum(lam)
        mean = aggregate.mean()
        se = aggregate.std(ddof=1) / np.sqrt(repetitions)
        half_width = student_t.ppf(0.975, repetitions - 1) * se
        print(
            f"{policy:10s}: class1={values[:, 0].mean():.6f}, "
            f"class2={values[:, 1].mean():.6f}, "
            f"overall={mean:.6f} +/- {half_width:.6f}"
        )
    return all_results


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    run_experiment()
