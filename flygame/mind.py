"""Мозг между чувствами и телом: входы -> частоты нейронов -> команда."""


class Mind:
    """Энкодеры, мозг и декодер одной мухи."""

    def __init__(self, brain, decoder, encoder, wind, odor):
        self.brain, self.decoder = brain, decoder
        self.encoder, self.wind, self.odor = encoder, wind, odor

    def sense(self, threats, boldness):
        """Зрение и поток воздуха -> (частоты входа, частоты одного ветра)."""
        wind_rates = self.wind.project(threats) * boldness
        rates = self.encoder.project(threats) * boldness + wind_rates
        return rates, wind_rates

    def smell(self, rates, concentration):
        """Добавить к входу запах еды."""
        return rates + self.odor.project({"fruit": concentration})

    def step(self, rates, duration_ms):
        """Прогнать мозг и прочитать команду с нисходящих нейронов."""
        self.brain.advance(rates, duration_ms=duration_ms)
        return self.decoder.decode(self.brain.output_rates_hz)

    def odor_level(self):
        """Уровень запаха с антеннальной доли (не с выхода)."""
        return self.odor.odor_level(self.brain.watch_rates_hz)

    def adapt(self, motor, dt):
        """Медленная подстройка декодера: перекос сторон и опора подъёма."""
        self.decoder.adapt_lateral(motor.lateral, dt)
        self.decoder.adapt_climb(self.brain.output_rates_hz, dt)
