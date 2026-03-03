use ordered_float::OrderedFloat;

pub fn compute_fractional_knapsack<I>(capacity: f64, items: I) -> f64
where
    I: Iterator<Item = (f64, f64)>,
{
    let mut items = items.collect::<Vec<_>>();

    // Sort items by value-to-weight ratio in descending order
    items.sort_by_key(|&(v, w)| OrderedFloat(w / v));

    compute_fractional_knapsack_sorted(capacity, items.into_iter())
}

pub fn compute_fractional_knapsack_sorted<I>(capacity: f64, sorted_items: I) -> f64
where
    I: Iterator<Item = (f64, f64)>,
{
    let mut total_value = 0.0;
    let mut remaining_capacity = capacity;

    for (value, weight) in sorted_items {
        if remaining_capacity == 0.0 {
            break;
        }

        if weight <= remaining_capacity {
            total_value += value;
            remaining_capacity -= weight;
        } else {
            total_value += value * (remaining_capacity / weight);
            remaining_capacity = 0.0;
        }
    }

    total_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fractional_knapsack() {
        let items = vec![(100.0, 20.0), (60.0, 10.0), (120.0, 30.0)];
        let capacity = 50.0;
        let result = compute_fractional_knapsack(capacity, items.into_iter());
        assert_eq!(result, 240.0);
    }

    #[test]
    fn test_compute_fractional_knapsack_sorted() {
        let items = vec![(60.0, 10.0), (100.0, 20.0), (120.0, 30.0)];
        let capacity = 50.0;
        let result = compute_fractional_knapsack_sorted(capacity, items.into_iter());
        assert_eq!(result, 240.0);
    }
}
