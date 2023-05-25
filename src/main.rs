use nnn::{Matrix, NeuralNetwork};

fn main() {
    let mut nn = NeuralNetwork::new(&[2, 2, 1]);

    let xs = Matrix::from(4, 2, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let ys = Matrix::from(4, 1, &[0.0, 1.0, 1.0, 0.0]);

    for epoch in 0..(1e4 as i32) {
        let output = nn.forward(&xs);

        let cost = output.sub(&ys).powi(2).mean();
        println!("epoch: {:7}, cost: {}", epoch + 1, cost);
        if cost < 1e-3 {
            break;
        }

        nn.update(&xs, &ys, 16.0);
    }

    for x1 in 0..=1 {
        for x2 in 0..=1 {
            println!(
                "{} {} = {:?}",
                x1,
                x2,
                nn.forward(&Matrix::from(1, 2, &[x1 as f64, x2 as f64]))
            );
        }
    }
}
