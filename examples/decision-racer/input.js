export class KeyboardInput {
  constructor() {
    this.leftPressed = false;
    this.rightPressed = false;
    this.onEnter = () => {};

    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        this.leftPressed = true;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        this.rightPressed = true;
      }
      if (event.key === "Enter") this.onEnter();
    });
  }

  consumeLaneMove() {
    if (this.leftPressed) {
      this.leftPressed = false;
      return -1;
    }
    if (this.rightPressed) {
      this.rightPressed = false;
      return 1;
    }
    return 0;
  }
}
