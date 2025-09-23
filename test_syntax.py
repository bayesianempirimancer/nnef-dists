class Test:
    def method(self):
        config = NetworkConfig(
            hidden_sizes=[32, 32],
            activation="swish",
            use_layer_norm=True,
            input_dim=12,
            output_dim=12
        )
        training_config = TrainingConfig(num_epochs=20, learning_rate=1e-2)
        return config, training_config
